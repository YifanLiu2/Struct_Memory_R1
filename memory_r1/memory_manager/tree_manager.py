"""
Tree-aware Memory Manager for Struct Memory-R1.

Extends the flat manager with tree-structure operations. The LLM outputs
JSON operations that include parent_id for ADD (to position new nodes in
the tree) and node_id for UPDATE/DELETE.
"""

import json
import re
import copy
import uuid
from typing import List, Dict, Optional, Tuple, Any

from memory_r1.memory_tree import MemoryTree, MemoryNode


def parse_tree_operations(llm_output: str) -> Optional[List[Dict]]:
    """Parse the LLM's JSON output into a list of tree operations.

    Expected format:
    {
        "memory": [
            {"id": "new_1", "parent_id": "root", "node_type": "MemoryEntry",
             "text": "...", "event": "ADD"},
            {"id": "existing_1", "text": "new content", "event": "UPDATE",
             "old_memory": "old content"},
            {"id": "existing_2", "event": "DELETE"},
            {"id": "existing_3", "text": "...", "event": "NONE"}
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


def validate_tree_operation(op: Dict) -> bool:
    """Validate a single tree operation."""
    if not isinstance(op, dict) or "event" not in op:
        return False
    event = op["event"].upper()
    if event not in ("ADD", "UPDATE", "DELETE", "NONE", "NOOP"):
        return False
    if event == "ADD" and ("parent_id" not in op or "text" not in op):
        return False
    if event == "UPDATE" and "text" not in op:
        return False
    return True


def apply_tree_operations(tree: MemoryTree,
                           operations: List[Dict]) -> Tuple[MemoryTree, Dict[str, int]]:
    """Apply a list of parsed operations to a MemoryTree.

    Returns the updated tree and operation stats.
    """
    stats = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0, "invalid": 0}

    for op in operations:
        if not validate_tree_operation(op):
            stats["invalid"] += 1
            continue

        event = op["event"].upper()

        if event in ("NONE", "NOOP"):
            stats["NONE"] += 1
            continue

        if event == "ADD":
            parent_id = op["parent_id"]
            node_id = op.get("id", f"auto_{uuid.uuid4().hex[:8]}")
            node_type = op.get("node_type", "MemoryEntry")
            text = op["text"]

            child = MemoryNode(
                node_id=node_id,
                node_type=node_type,
                attributes={"text": text},
            )
            success = tree.add_child(parent_id, child)
            if success:
                stats["ADD"] += 1
            else:
                if tree.root is not None:
                    tree.add_child(tree.root.node_id, child)
                    stats["ADD"] += 1
                else:
                    stats["invalid"] += 1

        elif event == "UPDATE":
            node_id = op.get("id", "")
            node = tree.get_node(node_id)
            if node is not None:
                node.attributes["text"] = op["text"]
                for k, v in op.items():
                    if k not in ("id", "text", "event", "old_memory", "parent_id", "node_type"):
                        node.attributes[k] = str(v)
                stats["UPDATE"] += 1
            else:
                stats["invalid"] += 1

        elif event == "DELETE":
            node_id = op.get("id", "")
            success = tree.remove_node(node_id)
            if success:
                stats["DELETE"] += 1
            else:
                stats["invalid"] += 1

    return tree, stats


class TreeMemoryManager:
    """High-level Memory Manager for tree-structured memory."""

    def __init__(self):
        pass

    def process(self, llm_output: str,
                tree: MemoryTree) -> Tuple[MemoryTree, Dict[str, int]]:
        """Parse LLM output and apply operations to the memory tree.

        Makes a deep copy of the tree before modification.
        """
        tree_copy = MemoryTree.from_json(tree.to_json())
        operations = parse_tree_operations(llm_output)
        if operations is None:
            return tree_copy, {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0, "invalid": 1}
        return apply_tree_operations(tree_copy, operations)

    def format_tree_for_prompt(self, tree: MemoryTree, max_depth: int = 5) -> str:
        """Render the tree as indented text for inclusion in the prompt."""
        return tree.to_text(max_depth=max_depth)

    def format_tree_with_ids(self, tree: MemoryTree, max_depth: int = 5) -> str:
        """Render the tree with node IDs visible (for the LLM to reference)."""
        lines = []
        self._render_node_with_id(tree.root, lines, depth=0, max_depth=max_depth)
        return "\n".join(lines)

    def _render_node_with_id(self, node: MemoryNode, lines: List[str],
                              depth: int, max_depth: int):
        if depth > max_depth:
            return
        indent = "  " * depth
        attr_str = " | ".join(f"{k}: {v}" for k, v in node.attributes.items())
        if attr_str:
            lines.append(f"{indent}[{node.node_type}] (id={node.node_id}) {attr_str}")
        else:
            lines.append(f"{indent}[{node.node_type}] (id={node.node_id})")
        for child in node.children:
            self._render_node_with_id(child, lines, depth + 1, max_depth)
