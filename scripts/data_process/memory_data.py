"""
Preprocess memory QA datasets to parquet format for RL training.

Generates training data in the same schema as Search-R1 (nq_search.py),
with prompts adapted for memory access instead of search.

Supports:
  - Semantic XPath domains (itinerary, todo, mealkit)
  - LoCoMo structured/flat
"""

import os
import json
import argparse
import datasets
import random

random.seed(42)


MEMORY_PROMPT_TEMPLATE = """Answer the given question using the provided memory. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you need to access memory, you can query by <memory> query </memory> and it will return relevant memory entries between <information> and </information>. \
You can query memory as many times as you want. \
If you find no further memory access needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


def make_memory_prefix(question: str) -> str:
    return MEMORY_PROMPT_TEMPLATE.format(question=question)


def process_semantic_xpath_domain(domain_path: str, data_source: str, split: str = "train"):
    """Process a Semantic XPath domain JSON file into training records."""
    with open(domain_path, "r") as f:
        data = json.load(f)

    qa_pairs = data["qa_pairs"]
    records = []

    for idx, qa in enumerate(qa_pairs):
        question = qa["question"].strip()
        if not question.endswith("?"):
            question += "?"

        answer = qa["answer"]
        if isinstance(answer, list):
            target = answer
        else:
            target = [str(answer)]

        record = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": make_memory_prefix(question)}],
            "ability": "memory-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": target},
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "question_type": qa.get("type", "unknown"),
            },
        }
        records.append(record)

    return records


def process_locomo_flat(locomo_path: str, split: str = "train"):
    """Process LoCoMo flat memory data into training records."""
    with open(locomo_path, "r") as f:
        data = json.load(f)

    records = []
    idx = 0
    dialogues = data if isinstance(data, list) else [data]

    for dialogue in dialogues:
        qa_pairs = dialogue.get("qa_pairs", [])
        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "")
            if not question:
                continue
            if not question.endswith("?"):
                question += "?"

            target = [answer] if isinstance(answer, str) else answer

            record = {
                "data_source": "locomo",
                "prompt": [{"role": "user", "content": make_memory_prefix(question)}],
                "ability": "memory-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": target},
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "dialogue_id": dialogue.get("dialogue_id", ""),
                },
            }
            records.append(record)
            idx += 1

    return records


def process_locomo_structured(locomo_path: str, split: str = "train"):
    """Process LoCoMo structured memory data into training records."""
    with open(locomo_path, "r") as f:
        data = json.load(f)

    records = []
    idx = 0
    dialogues = data if isinstance(data, list) else [data]

    for dialogue in dialogues:
        qa_pairs = dialogue.get("qa_pairs", [])
        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "")
            if not question:
                continue
            if not question.endswith("?"):
                question += "?"

            target = [answer] if isinstance(answer, str) else answer

            record = {
                "data_source": "locomo_structured",
                "prompt": [{"role": "user", "content": make_memory_prefix(question)}],
                "ability": "memory-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": target},
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "dialogue_id": dialogue.get("dialogue_id", ""),
                },
            }
            records.append(record)
            idx += 1

    return records


def build_dataset(records, split_ratio=0.8):
    """Split records into train/test and return as HuggingFace datasets."""
    random.shuffle(records)
    split_idx = int(len(records) * split_ratio)
    train_records = records[:split_idx]
    test_records = records[split_idx:]

    for i, r in enumerate(train_records):
        r["extra_info"]["split"] = "train"
        r["extra_info"]["index"] = i
    for i, r in enumerate(test_records):
        r["extra_info"]["split"] = "test"
        r["extra_info"]["index"] = i

    train_ds = datasets.Dataset.from_list(train_records)
    test_ds = datasets.Dataset.from_list(test_records)
    return train_ds, test_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./memory_r1/data",
                        help="Directory containing memory data JSONs")
    parser.add_argument("--output_dir", default="./data/memory_train",
                        help="Output directory for parquet files")
    parser.add_argument("--include_locomo", action="store_true",
                        help="Include LoCoMo data if available")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_records = []

    domain_configs = [
        ("itinerary.json", "itinerary"),
        ("todo.json", "todo"),
        ("mealkit.json", "mealkit"),
    ]

    for filename, source in domain_configs:
        filepath = os.path.join(args.data_dir, filename)
        if os.path.exists(filepath):
            records = process_semantic_xpath_domain(filepath, source)
            all_records.extend(records)
            print(f"Loaded {len(records)} records from {source}")

    if args.include_locomo:
        locomo_flat = os.path.join(args.data_dir, "locomo", "locomo_flat.json")
        locomo_struct = os.path.join(args.data_dir, "locomo", "locomo_structured.json")

        if os.path.exists(locomo_flat):
            records = process_locomo_flat(locomo_flat)
            all_records.extend(records)
            print(f"Loaded {len(records)} LoCoMo flat records")

        if os.path.exists(locomo_struct):
            records = process_locomo_structured(locomo_struct)
            all_records.extend(records)
            print(f"Loaded {len(records)} LoCoMo structured records")

    print(f"\nTotal records: {len(all_records)}")

    train_ds, test_ds = build_dataset(all_records)
    train_ds.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    test_ds.to_parquet(os.path.join(args.output_dir, "test.parquet"))

    print(f"Train: {len(train_ds)} samples -> {args.output_dir}/train.parquet")
    print(f"Test: {len(test_ds)} samples -> {args.output_dir}/test.parquet")
