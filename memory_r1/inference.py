"""
End-to-end inference pipeline for Memory-R1 and Struct Memory-R1.

Implements the full two-stage pipeline from the paper:
  Stage 1: Process dialogue turns through Memory Manager -> build memory bank
  Stage 2: For each question, retrieve top-k memories -> Answer Agent -> answer

Supports both flat (Memory-R1) and structured (Struct Memory-R1) memory.
"""

import os
import json
import argparse
import copy
from typing import List, Dict, Optional, Tuple, Any

from memory_r1.flat_memory import FlatMemoryStore
from memory_r1.memory_tree import MemoryTree, MemoryNode
from memory_r1.memory_manager.flat_manager import FlatMemoryManager
from memory_r1.memory_manager.tree_manager import TreeMemoryManager
from memory_r1.answer_agent.answer_agent import (
    AnswerAgent,
    format_flat_memory_context,
    format_tree_memory_context,
)
from memory_r1.evaluation import evaluate_predictions, evaluate_by_type

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# LLM inference helpers
# ---------------------------------------------------------------------------

def load_hf_model(model_path: str, device: str = "auto"):
    """Load a HuggingFace model and tokenizer."""
    assert HAS_TRANSFORMERS, "pip install transformers torch"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    return model, tokenizer


def generate_hf(model, tokenizer, prompt: str,
                max_new_tokens: int = 1024,
                temperature: float = 0.0) -> str:
    """Generate text using a HuggingFace model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_openai(client, prompt: str, system: str = "",
                    model: str = "gpt-4o-mini",
                    temperature: float = 0.0,
                    max_tokens: int = 1024) -> str:
    """Generate text using OpenAI API."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Stage 1: Memory Bank Construction via Memory Manager
# ---------------------------------------------------------------------------

def build_memory_bank_flat(
    dialogue_turns: List[Dict],
    generate_fn,
    mm_system_prompt: str = "",
    window: int = 50,
) -> List[Dict]:
    """Build a flat memory bank by running the Memory Manager on each turn.

    Args:
        dialogue_turns: list of {speaker, message, timestamp} dicts
        generate_fn: callable(prompt, system) -> str
        mm_system_prompt: system prompt for the Memory Manager
        window: max number of recent bank entries to show in prompt

    Returns:
        Final memory bank as list of {id, text} dicts
    """
    from memory_r1.memory_manager.prompts import (
        FLAT_MEMORY_MANAGER_SYSTEM,
        FLAT_MEMORY_MANAGER_USER,
        format_flat_memory_bank,
    )
    from memory_r1.data.data_construction import extract_facts_heuristic

    manager = FlatMemoryManager()
    bank: List[Dict] = []

    system = mm_system_prompt or FLAT_MEMORY_MANAGER_SYSTEM

    for turn in dialogue_turns:
        speaker = turn.get("speaker", "unknown")
        message = turn.get("message", turn.get("text", ""))
        if not message.strip():
            continue

        facts = extract_facts_heuristic(speaker, message)

        user_prompt = FLAT_MEMORY_MANAGER_USER.format(
            memory_bank=format_flat_memory_bank(bank[-window:]),
            extracted_facts=json.dumps(facts),
        )

        mm_output = generate_fn(user_prompt, system)

        bank, stats = manager.process(mm_output, bank)

    return bank


def build_memory_bank_tree(
    dialogue_turns: List[Dict],
    generate_fn,
    initial_tree: Optional[MemoryTree] = None,
    mm_system_prompt: str = "",
) -> MemoryTree:
    """Build a tree-structured memory bank by running the Tree Memory Manager.

    Args:
        dialogue_turns: list of {speaker, message, timestamp} dicts
        generate_fn: callable(prompt, system) -> str
        initial_tree: optional pre-existing tree to extend
        mm_system_prompt: system prompt for the Memory Manager

    Returns:
        Updated MemoryTree
    """
    from memory_r1.memory_manager.prompts import (
        TREE_MEMORY_MANAGER_SYSTEM,
        TREE_MEMORY_MANAGER_USER,
    )
    from memory_r1.data.data_construction import extract_facts_heuristic

    manager = TreeMemoryManager()

    if initial_tree is None:
        root = MemoryNode(node_id="root", node_type="Dialogue", attributes={})
        tree = MemoryTree(root)
    else:
        tree = MemoryTree.from_json(initial_tree.to_json())

    system = mm_system_prompt or TREE_MEMORY_MANAGER_SYSTEM

    for turn in dialogue_turns:
        speaker = turn.get("speaker", "unknown")
        message = turn.get("message", turn.get("text", ""))
        if not message.strip():
            continue

        facts = extract_facts_heuristic(speaker, message)

        tree_text = manager.format_tree_with_ids(tree, max_depth=5)
        user_prompt = TREE_MEMORY_MANAGER_USER.format(
            tree_structure=tree_text,
            extracted_facts=json.dumps(facts),
        )

        mm_output = generate_fn(user_prompt, system)
        tree, stats = manager.process(mm_output, tree)

    return tree


# ---------------------------------------------------------------------------
# Stage 2: Answer Generation via Answer Agent
# ---------------------------------------------------------------------------

def answer_questions_flat(
    questions: List[Dict],
    memory_bank: List[Dict],
    generate_fn,
    aa_system_prompt: str = "",
    topk: int = 60,
) -> List[Dict]:
    """Answer questions using the Answer Agent with flat memory retrieval.

    Args:
        questions: list of {question, answer, type} dicts
        memory_bank: flat memory bank
        generate_fn: callable(prompt, system) -> str
        aa_system_prompt: system prompt for the Answer Agent
        topk: number of memories to retrieve

    Returns:
        list of {question, gold_answer, predicted_answer, type} dicts
    """
    from memory_r1.answer_agent.prompts import ANSWER_AGENT_SYSTEM
    from memory_r1.data.data_construction import keyword_retrieve

    agent = AnswerAgent(memory_type="flat")
    system = aa_system_prompt or ANSWER_AGENT_SYSTEM

    results = []
    for qa in questions:
        question = qa.get("question", qa.get("q", ""))
        gold = qa.get("answer", qa.get("a", ""))
        qtype = qa.get("type", qa.get("category", ""))

        retrieved = keyword_retrieve(question, memory_bank, topk=topk)
        context = format_flat_memory_context(retrieved)
        prompt = f"{context}\n\nQuestion: {question}\n"

        output = generate_fn(prompt, system)
        parsed = agent.parse_output(output)

        results.append({
            "question": question,
            "gold_answer": gold,
            "predicted_answer": parsed["answer"] or "",
            "type": qtype,
            "raw_output": output,
            "selected_memories": parsed["selected_memories"],
        })

    return results


def answer_questions_tree(
    questions: List[Dict],
    tree: MemoryTree,
    generate_fn,
    aa_system_prompt: str = "",
    topk: int = 60,
) -> List[Dict]:
    """Answer questions using the Answer Agent with tree-structured memory."""
    from memory_r1.answer_agent.prompts import ANSWER_AGENT_SYSTEM_TREE

    agent = AnswerAgent(memory_type="structured")
    system = aa_system_prompt or ANSWER_AGENT_SYSTEM_TREE

    results = []
    for qa in questions:
        question = qa.get("question", qa.get("q", ""))
        gold = qa.get("answer", qa.get("a", ""))
        qtype = qa.get("type", qa.get("category", ""))

        matches = tree.keyword_search(question, topk=topk)
        retrieved = []
        for node, score in matches:
            subtree_text = tree.get_subtree_text(node, max_depth=3)
            retrieved.append({
                "path": node.path,
                "text": subtree_text,
                "timestamp": node.attributes.get("timestamp", ""),
            })

        context = format_tree_memory_context(retrieved)
        prompt = f"{context}\n\nQuestion: {question}\n"

        output = generate_fn(prompt, system)
        parsed = agent.parse_output(output)

        results.append({
            "question": question,
            "gold_answer": gold,
            "predicted_answer": parsed["answer"] or "",
            "type": qtype,
            "raw_output": output,
            "selected_memories": parsed["selected_memories"],
        })

    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dialogue_turns: List[Dict],
    questions: List[Dict],
    generate_fn,
    memory_type: str = "flat",
    mm_system: str = "",
    aa_system: str = "",
    topk: int = 60,
    initial_tree: Optional[MemoryTree] = None,
) -> Tuple[Any, List[Dict], Dict[str, float]]:
    """Run the full Memory-R1 pipeline: build bank, then answer questions.

    Returns:
        (memory_bank_or_tree, answer_results, metrics)
    """
    print(f"Stage 1: Building {memory_type} memory bank from {len(dialogue_turns)} turns...")

    if memory_type == "structured":
        memory = build_memory_bank_tree(
            dialogue_turns, generate_fn,
            initial_tree=initial_tree, mm_system_prompt=mm_system,
        )
        print(f"  Built tree with {len(memory)} nodes")
    else:
        memory = build_memory_bank_flat(
            dialogue_turns, generate_fn,
            mm_system_prompt=mm_system,
        )
        print(f"  Built flat bank with {len(memory)} entries")

    print(f"Stage 2: Answering {len(questions)} questions...")

    if memory_type == "structured":
        results = answer_questions_tree(
            questions, memory, generate_fn,
            aa_system_prompt=aa_system, topk=topk,
        )
    else:
        results = answer_questions_flat(
            questions, memory, generate_fn,
            aa_system_prompt=aa_system, topk=topk,
        )

    preds = [r["predicted_answer"] for r in results]
    golds = [r["gold_answer"] for r in results]
    qs = [r["question"] for r in results]

    metrics = evaluate_predictions(preds, golds, qs)
    print(f"  Results: {metrics}")

    types = [r.get("type", "") for r in results]
    if any(types):
        typed_metrics = evaluate_by_type(preds, golds, types, qs)
        for t, m in typed_metrics.items():
            print(f"  {t}: F1={m['F1']:.1f} BLEU1={m['BLEU1']:.1f}")

    return memory, results, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Memory-R1 inference pipeline")
    parser.add_argument("--mode", choices=["pipeline", "evaluate", "rebuild_banks"],
                        default="pipeline")
    parser.add_argument("--locomo_path", type=str, help="Path to LoCoMo JSON")
    parser.add_argument("--memory_type", choices=["flat", "structured"], default="flat")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HF model path or 'openai' for API")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--topk", type=int, default=60)
    parser.add_argument("--output_path", type=str, default="results.json")
    parser.add_argument("--use_judge", action="store_true")
    args = parser.parse_args()

    if args.model_path == "openai" or (args.model_path is None and HAS_OPENAI):
        assert HAS_OPENAI, "pip install openai"
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        def generate_fn(prompt, system=""):
            return generate_openai(client, prompt, system=system,
                                    model=args.openai_model)
        print(f"Using OpenAI API ({args.openai_model})")

    elif args.model_path:
        model, tokenizer = load_hf_model(args.model_path)

        def generate_fn(prompt, system=""):
            full = f"{system}\n\n{prompt}" if system else prompt
            return generate_hf(model, tokenizer, full)
        print(f"Using HF model: {args.model_path}")
    else:
        raise ValueError("Specify --model_path or set OPENAI_API_KEY")

    if args.mode == "pipeline" and args.locomo_path:
        from memory_r1.data.data_construction import load_locomo

        dialogues = load_locomo(args.locomo_path)
        print(f"Loaded {len(dialogues)} dialogues")

        all_results = []
        for d_idx, dialogue in enumerate(dialogues):
            turns = []
            for session in dialogue.get("sessions", []):
                if isinstance(session, dict) and "turns" in session:
                    turns.extend(session["turns"])
            questions = dialogue.get("qa_pairs", [])

            _, results, metrics = run_pipeline(
                turns, questions, generate_fn,
                memory_type=args.memory_type, topk=args.topk,
            )
            all_results.extend(results)

        with open(args.output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output_path}")

    elif args.mode == "evaluate":
        with open(args.output_path) as f:
            results = json.load(f)

        preds = [r["predicted_answer"] for r in results]
        golds = [r["gold_answer"] for r in results]
        qs = [r["question"] for r in results]

        metrics = evaluate_predictions(preds, golds, qs, use_judge=args.use_judge)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
