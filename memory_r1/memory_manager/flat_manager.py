"""
Flat Memory Manager for Memory-R1.

Parses LLM-generated JSON operations (ADD, UPDATE, DELETE, NOOP) and applies
them to a FlatMemoryStore. Faithful to the Memory-R1 paper (Yan et al., 2025).
"""

import json
import re
import copy
from typing import List, Dict, Optional, Tuple, Any

from memory_r1.flat_memory import FlatMemoryStore, MemoryEntry


def parse_memory_operations(llm_output: str) -> Optional[List[Dict]]:
    """Parse the LLM's JSON output into a list of memory operations.

    Expected format:
    {
        "memory": [
            {"id": "0", "text": "...", "event": "ADD|UPDATE|DELETE|NONE", ...}
        ]
    }
    """
    text = llm_output.strip()

    json_match = re.search(r'\{[\s\S]*"memory"[\s\S]*\}', text)
    if json_match:
        text = json_match.group(0)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        text = text.replace("'", '"')
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None

    if isinstance(parsed, dict) and "memory" in parsed:
        ops = parsed["memory"]
        if isinstance(ops, list):
            return ops
    return None


def validate_operation(op: Dict) -> bool:
    """Check that an operation dict has the required fields."""
    if not isinstance(op, dict):
        return False
    if "event" not in op:
        return False
    event = op["event"].upper()
    if event not in ("ADD", "UPDATE", "DELETE", "NONE", "NOOP"):
        return False
    if event in ("ADD", "UPDATE", "NONE", "NOOP") and "text" not in op:
        return False
    if event == "UPDATE" and "old_memory" not in op:
        return False
    return True


def apply_operations(store: FlatMemoryStore,
                     operations: List[Dict]) -> Tuple[FlatMemoryStore, Dict[str, int]]:
    """Apply a list of parsed operations to a FlatMemoryStore.

    Returns the updated store and a stats dict counting each operation type.
    """
    stats = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0, "invalid": 0}

    for op in operations:
        if not validate_operation(op):
            stats["invalid"] += 1
            continue

        event = op["event"].upper()
        if event in ("NONE", "NOOP"):
            stats["NONE"] += 1
            continue

        if event == "ADD":
            store.add(text=op["text"])
            stats["ADD"] += 1

        elif event == "UPDATE":
            entry_id = op.get("id", "")
            success = store.update(entry_id, op["text"])
            if success:
                stats["UPDATE"] += 1
            else:
                store.add(text=op["text"])
                stats["ADD"] += 1

        elif event == "DELETE":
            entry_id = op.get("id", "")
            success = store.delete(entry_id)
            if success:
                stats["DELETE"] += 1
            else:
                stats["invalid"] += 1

    return store, stats


def apply_operations_to_bank(bank: List[Dict],
                              operations: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """Apply operations to a raw memory bank (list of {id, text} dicts).

    This variant works directly on the JSON representation used in training data,
    without requiring a full FlatMemoryStore instance.
    """
    bank = copy.deepcopy(bank)
    bank_by_id = {e["id"]: e for e in bank}
    stats = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0, "invalid": 0}
    next_id = max((int(e["id"]) for e in bank), default=-1) + 1

    for op in operations:
        if not validate_operation(op):
            stats["invalid"] += 1
            continue

        event = op["event"].upper()
        if event in ("NONE", "NOOP"):
            stats["NONE"] += 1
            continue

        if event == "ADD":
            new_id = op.get("id", str(next_id))
            bank.append({"id": new_id, "text": op["text"]})
            bank_by_id[new_id] = bank[-1]
            next_id = max(next_id, int(new_id) + 1) if new_id.isdigit() else next_id + 1
            stats["ADD"] += 1

        elif event == "UPDATE":
            eid = op.get("id", "")
            if eid in bank_by_id:
                bank_by_id[eid]["text"] = op["text"]
                stats["UPDATE"] += 1
            else:
                bank.append({"id": str(next_id), "text": op["text"]})
                next_id += 1
                stats["ADD"] += 1

        elif event == "DELETE":
            eid = op.get("id", "")
            if eid in bank_by_id:
                bank = [e for e in bank if e["id"] != eid]
                del bank_by_id[eid]
                stats["DELETE"] += 1
            else:
                stats["invalid"] += 1

    return bank, stats


class FlatMemoryManager:
    """High-level Memory Manager that wraps parsing + application for flat memory."""

    def __init__(self):
        pass

    def process(self, llm_output: str,
                current_bank: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
        """Parse LLM output and apply operations to the memory bank.

        Args:
            llm_output: raw text output from the Memory Manager LLM
            current_bank: list of {id, text} dicts

        Returns:
            updated_bank, operation_stats
        """
        operations = parse_memory_operations(llm_output)
        if operations is None:
            return current_bank, {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0, "invalid": 1}
        return apply_operations_to_bank(current_bank, operations)

    def format_bank_for_prompt(self, bank: List[Dict]) -> str:
        """Format memory bank as JSON string for the prompt."""
        return json.dumps(bank, indent=2) if bank else "[]"
