"""
LoCoMo Evaluation Runner - Evaluate Semantic XPath pipeline on LoCoMo questions.

Runs questions from questions.xml through the pipeline, then evaluates answers:
- Non-adversarial: LLM-as-judge comparing system vs ground truth
- Adversarial: keyword matching (checks if system correctly says "no info")

Usage:
    python -m experiment.locomo_infra.locomo_eval --conv conv-26
    python -m experiment.locomo_infra.locomo_eval --all
    python -m experiment.locomo_infra.locomo_eval --all --categories adversarial
    python -m experiment.locomo_infra.locomo_eval --conv conv-26 --max-questions 10
"""

import json
import yaml
import time
import re
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.semantic_xpath_pipeline import SemanticXPathPipeline
from client import OpenAIClient
from experiment.locomo_infra.prompts import ANSWER_PROMPT
from experiment.locomo_infra.locomo_metrics import evaluate_answer


# Project root (LLM-VM/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# LoCoMo infra directory (where this file lives)
LOCOMO_INFRA_DIR = Path(__file__).parent

# All LoCoMo conversations available for evaluation
ALL_CONVERSATIONS = [
    "conv-26", "conv-30", "conv-41", "conv-42", "conv-43",
    "conv-44", "conv-47", "conv-48", "conv-49", "conv-50",
]


def load_questions(conv_id: str) -> List[Dict[str, Any]]:
    """Load questions for a specific conversation from questions.xml (JSON format)."""
    questions_path = PROJECT_ROOT.parent / "locomo_structured_data" / "questions.xml"
    
    with open(questions_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    
    if conv_id not in all_questions:
        available = list(all_questions.keys())
        raise ValueError(f"Conversation '{conv_id}' not found. Available: {available}")
    
    return all_questions[conv_id]["questions"]


def extract_retrieved_context(result: Dict[str, Any], session_tree: Optional[Path] = None) -> str:
    """Extract retrieved observation content from pipeline result for the answer agent.
    
    Selected nodes from the read handler contain:
    - For Observation nodes: type, text, source_dia_id, tree_path, reasoning
    - For Session nodes: type, attributes (session_id, datetime, summary), tree_path, reasoning
    """
    selected = result.get("selected_nodes", [])
    if not selected:
        if result.get("error"):
            return f"Error: {result['error']}"
        return ""
    
    parts = []
    
    # Load XML tree to resolve session datetimes for Observations
    root_el = None
    if session_tree and session_tree.exists():
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(session_tree)
            root_el = tree.getroot()
        except Exception:
            pass

    for i, node in enumerate(selected, 1):
        tree_path = node.get("tree_path", "")
        
        lines = [f"--- Retrieved Item {i} ---"]
        lines.append(f"Path: {tree_path}")
        
        # Extract Session datetime for Observation nodes
        if root_el is not None and node.get("type", "") == "Observation" and "Session " in tree_path:
            try:
                path_parts = [p.strip() for p in tree_path.split(">")]
                
                session_idx = -1
                for idx, p in enumerate(path_parts):
                    if p.startswith("Session "):
                        session_idx = idx
                        break
                
                if session_idx > 0:
                    speaker_name = path_parts[session_idx - 1]
                    session_match = re.search(r'Session\s+(\d+)', path_parts[session_idx])
                    
                    if speaker_name and session_match:
                        session_id = session_match.group(1).strip()
                        
                        xpath = f".//Speaker[@name='{speaker_name}']/Session[@session_id='{session_id}']"
                        session_node = root_el.find(xpath)
                        if session_node is not None:
                            dt = session_node.get("datetime")
                            if dt:
                                lines.append(f"[Temporal Context] Session Datetime: {dt}")
            except Exception:
                pass
        
        # Include all fields from the node
        for key, val in node.items():
            if key in ("tree_path", "reasoning"):
                continue
            if key == "attributes" and isinstance(val, dict):
                for attr_k, attr_v in val.items():
                    lines.append(f"{attr_k}: {attr_v}")
            elif key == "children" and isinstance(val, list):
                for child in val:
                    if isinstance(child, dict):
                        child_type = child.get("type", "")
                        child_text = child.get("text", "")
                        if child_text:
                            lines.append(f"{child_type}: {child_text}")
                        for ck, cv in child.items():
                            if ck not in ("type", "text", "children") and cv:
                                lines.append(f"  {ck}: {cv}")
            elif val and key != "type":
                lines.append(f"{key}: {val}")
        
        reasoning = node.get("reasoning", "")
        if reasoning:
            lines.append(f"Reasoning: {reasoning}")
        
        parts.append("\n".join(lines))
    
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    context: str,
    client: Any
) -> str:
    """Answer agent: given question + retrieved observations, produce a concise answer."""
    if not context:
        return "No relevant information found"
    
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    
    try:
        answer = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256
        )
        return answer.strip()
    except Exception as e:
        return f"Answer generation error: {e}"


def run_evaluation(
    conv_id: str = "conv-26",
    categories: Optional[List[str]] = None,
    max_questions: Optional[int] = None,
    config_path: Optional[str] = None,
    prune_top_k: Optional[int] = None,
    start_index: Optional[int] = None,
    out_dir: str = "",
):
    """
    Run evaluation: pipeline answers + scoring (LLM judge or keyword match).
    
    Args:
        conv_id: Conversation ID to evaluate
        categories: Optional list of categories to filter (single_hop, temporal, etc.)
        max_questions: Optional max number of questions to evaluate
        config_path: Path to experiment config YAML
        prune_top_k: Top-K threshold for semantic pruning
    """
    # Load config (default to locomo_infra/conversation_experiment.yaml)
    if config_path is None:
        config_file = LOCOMO_INFRA_DIR / "conversation_experiment.yaml"
    else:
        config_file = PROJECT_ROOT / config_path
    
    with open(config_file, "r") as f:
        experiment_config = yaml.safe_load(f)
    app_config = experiment_config.get("config", {})
    
    # Override active_data to match conv_id
    app_config["active_data"] = conv_id
    
    # Inject Top-K Pruning setting
    if prune_top_k is not None:
        if "xpath_executor" not in app_config:
            app_config["xpath_executor"] = {}
        app_config["xpath_executor"]["prune_top_k_at_each_step"] = prune_top_k
    
    # Load questions
    questions = load_questions(conv_id)
    
    # Filter by category if specified
    if categories:
        questions = [q for q in questions if q["category"] in categories]
        
    # Start index filter
    if start_index is not None:
        questions = [q for q in questions if q.get("qa_index", 0) >= start_index]
    
    # Limit questions
    if max_questions:
        questions = questions[:max_questions]
    
    print(f"\n{'#'*60}")
    print(f"# LoCoMo Evaluation: {conv_id}")
    print(f"# Questions: {len(questions)}")
    if categories:
        print(f"# Categories: {', '.join(categories)}")
    print(f"{'#'*60}")
    
    # Setup output directory — results go to experiment/locomo/[out_dir]/
    base_dir = PROJECT_ROOT / "experiment" / "locomo"
    if out_dir:
        base_dir = base_dir / out_dir
    output_dir = base_dir / f"{conv_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    from pipeline_execution.semantic_xpath_execution import get_data_path
    source_tree = get_data_path(config=app_config)
    
    import copy
    judge_config = copy.deepcopy(app_config)
    if "openai" not in judge_config:
        judge_config["openai"] = {}
    judge_config["openai"]["model"] = "gpt-4o"
    judge_config["openai"]["api_key"] = "${OPENAI_API_KEY}"
    judge_config["openai"].pop("base_url", None)
    
    # Initialize LLM client for judging (used for non-adversarial)
    judge_client = OpenAIClient(judge_config)
    
    # Results tracking
    results = []
    category_stats = {}
    
    total_start = time.perf_counter()
    
    for i, q in enumerate(questions):
        qa_index = q["qa_index"]
        question = q["question"]
        ground_truth = str(q["answer"])
        category = q["category"]
        
        print(f"\n  [{i+1}/{len(questions)}] Q{qa_index} ({category}): {question[:60]}...")
        
        # Create per-question directory for traces
        q_name = re.sub(r'[^\w\s-]', '', question[:40])
        q_name = re.sub(r'\s+', '_', q_name).strip('_')
        query_dir = output_dir / f"Q{qa_index:03d}_{q_name}"
        query_dir.mkdir(parents=True, exist_ok=True)
        
        # Create fresh pipeline for each question
        session_tree = query_dir / "tree.xml"
        shutil.copy2(source_tree, session_tree)
        pipeline = SemanticXPathPipeline(tree_path=session_tree, config=app_config)
        pipeline.set_traces_path(query_dir)
        
        # Run pipeline
        start = time.perf_counter()
        try:
            result = pipeline.process_request(question)
            exec_time = (time.perf_counter() - start) * 1000
            success = result.get("success", False)
        except Exception as e:
            exec_time = (time.perf_counter() - start) * 1000
            success = False
            result = {}
        
        # Extract retrieved observation content
        retrieved_context = extract_retrieved_context(result, session_tree)
        
        # Answer Agent: generate concise answer from retrieved observations
        system_answer = generate_answer(question, retrieved_context, judge_client)
        
        # Evaluate the answer (adversarial → keyword match, others → LLM judge)
        judgment = evaluate_answer(
            category=category,
            system_answer=system_answer,
            ground_truth=ground_truth,
            question=question,
            client=judge_client,
        )
        verdict = judgment.get("verdict", "ERROR")
        score = judgment.get("score", 0.0)
        
        # Save answer agent + eval traces
        traces_dir = query_dir / "reasoning_traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        answer_trace = {
            "question": question,
            "retrieved_context": retrieved_context,
            "system_answer": system_answer,
            "selected_nodes": [n.get("tree_path", "") for n in result.get("selected_nodes", [])],
        }
        with open(traces_dir / f"answer_agent_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(answer_trace, f, indent=2, ensure_ascii=False)
        
        eval_trace = {
            "question": question,
            "ground_truth": ground_truth,
            "system_answer": system_answer,
            "category": category,
            "eval_method": "keyword_match" if category == "adversarial" else "llm_judge",
            "verdict": verdict,
            "score": score,
            "f1": judgment.get("f1", 0.0),
            "bleu_1": judgment.get("bleu_1", 0.0),
            "reasoning": judgment.get("reasoning", ""),
        }
        with open(traces_dir / f"eval_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(eval_trace, f, indent=2, ensure_ascii=False)
        
        print(f"    Answer: {system_answer[:80]}")
        print(f"    Truth:  {ground_truth[:80]}")
        eval_method_label = "keyword" if category == "adversarial" else "judge"
        print(f"    Verdict: {verdict} [{eval_method_label}] ({exec_time:.0f}ms)")
        
        # Record result
        entry = {
            "qa_index": qa_index,
            "question": question,
            "category": category,
            "ground_truth": ground_truth,
            "system_answer": system_answer,
            "retrieved_context": retrieved_context[:500],
            "verdict": verdict,
            "score": score,
            "eval_method": "keyword_match" if category == "adversarial" else "llm_judge",
            "eval_reasoning": judgment.get("reasoning", ""),
            "f1": judgment.get("f1", 0.0),
            "bleu_1": judgment.get("bleu_1", 0.0),
            "execution_time_ms": round(exec_time, 2),
            "pipeline_success": success,
            "evidence": q.get("evidence", []),
            "xpath": result.get("xpath_query", ""),
            "selected_nodes": [n.get("tree_path", "") for n in result.get("selected_nodes", [])]
        }
        results.append(entry)
        
        # Update category stats
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "wrong": 0, "error": 0, "total": 0, "f1_sum": 0.0, "bleu_1_sum": 0.0}
        category_stats[category]["total"] += 1
        category_stats[category][verdict.lower()] = category_stats[category].get(verdict.lower(), 0) + 1
        category_stats[category]["f1_sum"] = category_stats[category].get("f1_sum", 0.0) + judgment.get("f1", 0.0)
        category_stats[category]["bleu_1_sum"] = category_stats[category].get("bleu_1_sum", 0.0) + judgment.get("bleu_1", 0.0)
    
    total_time = (time.perf_counter() - total_start)
    
    # Calculate overall stats
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "CORRECT")
    wrong = sum(1 for r in results if r["verdict"] == "WRONG")
    errors = sum(1 for r in results if r["verdict"] == "ERROR")
    
    # Build summary
    summary = {
        "conversation_id": conv_id,
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "categories_evaluated": categories or "all",
        "overall": {
            "correct": correct,
            "wrong": wrong,
            "error": errors,
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
        },
        "by_category": {},
        "total_time_seconds": round(total_time, 1)
    }
    
    for cat, stats in category_stats.items():
        cat_total = stats["total"]
        f1_avg = stats.get("f1_sum", 0.0) / cat_total if cat_total > 0 else 0
        bleu_1_avg = stats.get("bleu_1_sum", 0.0) / cat_total if cat_total > 0 else 0
        
        # Create a copy so we can pop the internal sums
        clean_stats = {k: v for k, v in stats.items() if not k.endswith("_sum")}
        
        summary["by_category"][cat] = {
            **clean_stats,
            "accuracy": round(stats["correct"] / cat_total * 100, 1) if cat_total > 0 else 0,
            "f1": round(f1_avg, 3),
            "bleu_1": round(bleu_1_avg, 3),
        }
    
    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Complete: {conv_id}")
    print(f"{'='*60}")
    print(f"  Total: {total} questions in {total_time:.1f}s")
    print(f"  Correct:   {correct}/{total} ({correct/total*100:.1f}%)" if total else "")
    print(f"  Wrong:     {wrong}/{total} ({wrong/total*100:.1f}%)" if total else "")
    if errors:
        print(f"  Errors:    {errors}/{total}")
    
    print(f"\n  By Category:")
    for cat, stats in sorted(summary["by_category"].items()):
        c = stats["correct"]
        t = stats["total"]
        f1 = stats.get("f1", 0.0)
        bleu = stats.get("bleu_1", 0.0)
        eval_note = " [keyword]" if cat == "adversarial" else " [judge]"
        print(f"    {cat:15s}: {c}/{t} correct ({stats['accuracy']}%) | F1: {f1:.3f} BLEU-1: {bleu:.3f}{eval_note}")
    
    print(f"\n  Results saved to: {results_path}")
    
    return summary


def run_all_evaluations(
    conversations: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    max_questions: Optional[int] = None,
    config_path: Optional[str] = None,
    prune_top_k: Optional[int] = None,
    start_index: Optional[int] = None,
    out_dir: str = "",
):
    """
    Run evaluation across multiple conversations and produce aggregated results.
    
    Args:
        conversations: List of conv IDs to evaluate (default: ALL_CONVERSATIONS)
        categories: Optional category filter
        max_questions: Optional per-conversation question limit
        config_path: Path to experiment config YAML
        prune_top_k: Top-K threshold for semantic pruning
    
    Returns:
        Dict with per-conversation summaries and aggregated results.
    """
    conv_ids = conversations or ALL_CONVERSATIONS
    
    print(f"\n{'#'*60}")
    print(f"# LoCoMo Multi-Conversation Evaluation")
    print(f"# Conversations: {', '.join(conv_ids)}")
    if categories:
        print(f"# Categories: {', '.join(categories)}")
    print(f"{'#'*60}")
    
    all_start = time.perf_counter()
    per_conv_summaries = {}
    
    for conv_id in conv_ids:
        print(f"\n{'='*60}")
        print(f"  Starting: {conv_id}")
        print(f"{'='*60}")
        try:
            summary = run_evaluation(
                conv_id=conv_id,
                categories=categories,
                max_questions=max_questions,
                config_path=config_path,
                prune_top_k=prune_top_k,
                start_index=start_index,
                out_dir=out_dir,
            )
            per_conv_summaries[conv_id] = summary
        except Exception as e:
            print(f"  ERROR evaluating {conv_id}: {e}")
            per_conv_summaries[conv_id] = {"error": str(e)}
    
    all_time = time.perf_counter() - all_start
    
    # --- Aggregate across conversations ---
    agg_category_stats = {}
    agg_total = 0
    agg_correct = 0
    agg_wrong = 0
    agg_error = 0
    
    for conv_id, summary in per_conv_summaries.items():
        if "error" in summary:
            continue
        overall = summary.get("overall", {})
        agg_total += summary.get("total_questions", 0)
        agg_correct += overall.get("correct", 0)
        agg_wrong += overall.get("wrong", 0)
        agg_error += overall.get("error", 0)
        
        for cat, stats in summary.get("by_category", {}).items():
            if cat not in agg_category_stats:
                agg_category_stats[cat] = {"correct": 0, "wrong": 0, "error": 0, "total": 0}
            agg_category_stats[cat]["correct"] += stats.get("correct", 0)
            agg_category_stats[cat]["wrong"] += stats.get("wrong", 0)
            agg_category_stats[cat]["error"] += stats.get("error", 0)
            agg_category_stats[cat]["total"] += stats.get("total", 0)
    
    # Compute aggregated accuracies
    for cat in agg_category_stats:
        t = agg_category_stats[cat]["total"]
        agg_category_stats[cat]["accuracy"] = round(
            agg_category_stats[cat]["correct"] / t * 100, 1
        ) if t > 0 else 0
    
    aggregated = {
        "conversations_evaluated": [c for c in conv_ids if "error" not in per_conv_summaries.get(c, {})],
        "conversations_failed": [c for c in conv_ids if "error" in per_conv_summaries.get(c, {})],
        "timestamp": datetime.now().isoformat(),
        "categories_evaluated": categories or "all",
        "total_questions": agg_total,
        "overall": {
            "correct": agg_correct,
            "wrong": agg_wrong,
            "error": agg_error,
            "accuracy": round(agg_correct / agg_total * 100, 1) if agg_total > 0 else 0,
        },
        "by_category": agg_category_stats,
        "total_time_seconds": round(all_time, 1),
    }
    
    # Save aggregated results
    base_dir = PROJECT_ROOT / "experiment" / "locomo"
    if out_dir:
        base_dir = base_dir / out_dir
    agg_dir = base_dir / f"aggregated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    agg_dir.mkdir(parents=True, exist_ok=True)
    
    agg_results = {
        "aggregated": aggregated,
        "per_conversation": per_conv_summaries,
    }
    agg_path = agg_dir / "aggregated_results.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg_results, f, indent=2, ensure_ascii=False)
    
    # Print aggregated summary
    print(f"\n\n{'*'*60}")
    print(f"* AGGREGATED RESULTS ({len(aggregated['conversations_evaluated'])} conversations)")
    print(f"{'*'*60}")
    print(f"  Total: {agg_total} questions in {all_time:.1f}s")
    if agg_total > 0:
        print(f"  Correct:   {agg_correct}/{agg_total} ({agg_correct/agg_total*100:.1f}%)")
        print(f"  Wrong:     {agg_wrong}/{agg_total} ({agg_wrong/agg_total*100:.1f}%)")
    if agg_error:
        print(f"  Errors:    {agg_error}/{agg_total}")
    
    print(f"\n  By Category (aggregated):")
    for cat, stats in sorted(agg_category_stats.items()):
        c = stats["correct"]
        t = stats["total"]
        eval_note = " [keyword]" if cat == "adversarial" else " [judge]"
        print(f"    {cat:15s}: {c}/{t} correct ({stats['accuracy']}%){eval_note}")
    
    print(f"\n  Per Conversation:")
    for conv_id, summary in per_conv_summaries.items():
        if "error" in summary:
            print(f"    {conv_id:10s}: ERROR - {summary['error']}")
        else:
            overall = summary.get("overall", {})
            t = summary.get("total_questions", 0)
            c = overall.get("correct", 0)
            acc = overall.get("accuracy", 0)
            print(f"    {conv_id:10s}: {c}/{t} correct ({acc}%)")
    
    print(f"\n  Aggregated results saved to: {agg_path}")
    
    return agg_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Semantic XPath pipeline on LoCoMo questions")
    parser.add_argument("--conv", type=str, default="conv-26", help="Conversation ID (default: conv-26)")
    parser.add_argument("--all", action="store_true", dest="run_all", help="Run evaluation on all 10 LoCoMo conversations")
    parser.add_argument("--categories", nargs="+", help="Filter by categories (single_hop, temporal, open_domain, multi_hop, adversarial)")
    parser.add_argument("--start-index", type=int, default=None, help="Start evaluating from this qa_index")
    parser.add_argument("--max-questions", type=int, help="Max questions to evaluate per conversation")
    parser.add_argument("--config", type=str, default=None, help="Experiment config path (default: locomo_infra/conversation_experiment.yaml)")
    parser.add_argument("--prune-top-k", type=int, default=5, help="Top-K threshold for semantic pruning (default: 5)")
    parser.add_argument("--out-dir", type=str, default="", help="Subdirectory under experiment/locomo to output results")
    
    args = parser.parse_args()
    
    if args.run_all:
        run_all_evaluations(
            categories=args.categories,
            max_questions=args.max_questions,
            config_path=args.config,
            prune_top_k=args.prune_top_k,
            start_index=args.start_index,
            out_dir=args.out_dir,
        )
    else:
        run_evaluation(
            conv_id=args.conv,
            categories=args.categories,
            max_questions=args.max_questions,
            config_path=args.config,
            prune_top_k=args.prune_top_k,
            start_index=args.start_index,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
