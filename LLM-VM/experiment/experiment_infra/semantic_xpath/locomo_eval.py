"""
LoCoMo Evaluation Runner - Evaluate Semantic XPath pipeline on LoCoMo questions.

Runs questions from questions.xml through the pipeline, then uses LLM-as-judge
to compare system answers against ground truth.

Usage:
    python -m experiment.experiment_infra.semantic_xpath.locomo_eval --conv conv-26
    python -m experiment.experiment_infra.semantic_xpath.locomo_eval --conv conv-26 --categories single_hop temporal
    python -m experiment.experiment_infra.semantic_xpath.locomo_eval --conv conv-26 --max-questions 10
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pipeline.semantic_xpath_pipeline import SemanticXPathPipeline
from client import OpenAIClient


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Answer Agent prompt (Memory-R1 style)
ANSWER_PROMPT = """You are an Answer Agent. Given a question and retrieved memory observations, provide a concise, direct answer.

Question: {question}

Retrieved Observations:
{context}

Rules:
- Answer the question directly and concisely based ONLY on the provided observations.
- If the observations contain the answer, state it clearly in 1-2 sentences.
- If the observations do not contain enough information to answer, say "Insufficient information."
- Do NOT add speculation beyond what the observations state.
- For temporal questions, use specific dates/times from the observation metadata when available.

Answer:"""

# LLM-as-Judge prompt (from LoCoMo paper)
JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer,
which you will score as CORRECT or WRONG.
The point of the question is to ask about something one user should know about the other user based on their
prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be longer, but you should be generous with your grading — as long as it touches
on the same topic as the gold answer, it should be counted as CORRECT.
For time-related questions, the gold answer will be a specific date, month, or year. The generated answer
might include relative references (e.g., "last Tuesday"), but you should be generous — if it refers to
the same time period as the gold answer, mark it CORRECT, even if the format differs (e.g., "May 7th" vs.
"7 May").
Now it's time for the real question:
Question: {question}
Gold answer: {ground_truth}
Generated answer: {system_answer}
First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
Return the label in JSON format with the key as "label"."""


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
                # e.g. Root > Conversation_Version 1 > Caroline > Session 1 > Observation 1
                path_parts = [p.strip() for p in tree_path.split(">")]
                
                # Find the index of the part containing "Session "
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
                        
                        # Find the corresponding Session node in the XML
                        xpath = f".//Speaker[@name='{speaker_name}']/Session[@session_id='{session_id}']"
                        session_node = root_el.find(xpath)
                        if session_node is not None:
                            dt = session_node.get("datetime")
                            if dt:
                                lines.append(f"[Temporal Context] Session Datetime: {dt}")
            except Exception:
                pass
        
        
        # Include all fields from the node (type, text, source_dia_id, attributes, etc.)
        for key, val in node.items():
            if key in ("tree_path", "reasoning"):
                continue  # handled separately
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
        
        # Include the read handler's reasoning (describes what the LLM understood)
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


def judge_answer(
    question: str,
    ground_truth: str,
    system_answer: str,
    client: Any
) -> Dict[str, Any]:
    """Use LLM to judge if system answer matches ground truth (LoCoMo paper style)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        system_answer=system_answer
    )
    
    try:
        # client.chat() returns a plain string
        text = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256
        )
        
        # Try to extract JSON with "label" key
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            judgment = json.loads(json_match.group())
            # Normalize: the paper uses "label" key
            label = judgment.get("label", judgment.get("verdict", "")).upper()
            # Extract reasoning from the text before the JSON
            reasoning = text[:json_match.start()].strip()
            return {"verdict": label, "reasoning": reasoning}
        
        # Fallback: look for CORRECT or WRONG in raw text
        text_upper = text.upper()
        if "CORRECT" in text_upper and "WRONG" not in text_upper:
            return {"verdict": "CORRECT", "reasoning": text.strip()}
        elif "WRONG" in text_upper and "CORRECT" not in text_upper:
            return {"verdict": "WRONG", "reasoning": text.strip()}
        else:
            return {"verdict": "ERROR", "reasoning": f"Could not parse judge response: {text}"}
    except Exception as e:
        return {"verdict": "ERROR", "reasoning": str(e)}


def run_evaluation(
    conv_id: str = "conv-26",
    categories: Optional[List[str]] = None,
    max_questions: Optional[int] = None,
    config_path: str = "conversation_experiment.yaml",
    prune_top_k: Optional[int] = None,
):
    """
    Run evaluation: pipeline answers + LLM-as-judge scoring.
    
    Args:
        conv_id: Conversation ID to evaluate
        categories: Optional list of categories to filter (single_hop, temporal, etc.)
        max_questions: Optional max number of questions to evaluate
        config_path: Path to experiment config YAML
    """
    # Load config
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
    
    # Limit questions
    if max_questions:
        questions = questions[:max_questions]
    
    print(f"\n{'#'*60}")
    print(f"# LoCoMo Evaluation: {conv_id}")
    print(f"# Questions: {len(questions)}")
    if categories:
        print(f"# Categories: {', '.join(categories)}")
    print(f"{'#'*60}")
    
    # Setup output directory
    output_dir = PROJECT_ROOT / "experiment" / "experiment_result" / "locomo_eval" / f"{conv_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    from pipeline_execution.semantic_xpath_execution import get_data_path
    source_tree = get_data_path(config=app_config)
    
    # Initialize LLM client for judging
    judge_client = OpenAIClient(app_config)
    
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
        
        # Judge the answer
        judgment = judge_answer(question, ground_truth, system_answer, judge_client)
        verdict = judgment.get("verdict", "ERROR")
        
        # Save answer agent + judge traces
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
        
        judge_trace = {
            "question": question,
            "ground_truth": ground_truth,
            "system_answer": system_answer,
            "verdict": verdict,
            "judge_reasoning": judgment.get("reasoning", ""),
        }
        with open(traces_dir / f"judge_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(judge_trace, f, indent=2, ensure_ascii=False)
        
        print(f"    Answer: {system_answer[:80]}")
        print(f"    Truth:  {ground_truth[:80]}")
        print(f"    Verdict: {verdict} ({exec_time:.0f}ms)")
        
        # Record result
        entry = {
            "qa_index": qa_index,
            "question": question,
            "category": category,
            "ground_truth": ground_truth,
            "system_answer": system_answer,
            "retrieved_context": retrieved_context[:500],  # truncate for readability
            "verdict": verdict,
            "judge_reasoning": judgment.get("reasoning", ""),
            "execution_time_ms": round(exec_time, 2),
            "pipeline_success": success,
            "evidence": q.get("evidence", []),
            "xpath": result.get("xpath_query", ""),
            "selected_nodes": [n.get("tree_path", "") for n in result.get("selected_nodes", [])]
        }
        results.append(entry)
        
        # Update category stats
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "wrong": 0, "error": 0, "total": 0}
        category_stats[category]["total"] += 1
        category_stats[category][verdict.lower()] = category_stats[category].get(verdict.lower(), 0) + 1
    
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
        summary["by_category"][cat] = {
            **stats,
            "accuracy": round(stats["correct"] / cat_total * 100, 1) if cat_total > 0 else 0,
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
        print(f"    {cat:15s}: {c}/{t} correct ({stats['accuracy']}%)")
    
    print(f"\n  Results saved to: {results_path}")
    
    return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Semantic XPath pipeline on LoCoMo questions")
    parser.add_argument("--conv", type=str, default="conv-26", help="Conversation ID (default: conv-26)")
    parser.add_argument("--categories", nargs="+", help="Filter by categories (single_hop, temporal, open_domain, multi_hop, adversarial)")
    parser.add_argument("--max-questions", type=int, help="Max questions to evaluate")
    parser.add_argument("--config", type=str, default="conversation_experiment.yaml", help="Experiment config path")
    parser.add_argument("--prune-top-k", type=int, default=5, help="Top-K threshold for semantic pruning (default: 3)")
    
    args = parser.parse_args()
    run_evaluation(
        conv_id=args.conv,
        categories=args.categories,
        max_questions=args.max_questions,
        config_path=args.config,
        prune_top_k=args.prune_top_k
    )


if __name__ == "__main__":
    main()
