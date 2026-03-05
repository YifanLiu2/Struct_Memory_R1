"""
Tree-structured memory data model for Structured Memory-R1.

Represents memory as a rooted tree M_t = (V_t, E_t, r) where each node
corresponds to a memory unit and edges encode hierarchical relations.
Supports both compositional hierarchies (e.g., Itinerary -> Day -> POI)
and session-based hierarchies (e.g., Dialogue -> Session -> Memory Entry).
"""

import json
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class MemoryNode:
    node_id: str
    node_type: str
    attributes: Dict[str, str] = field(default_factory=dict)
    children: List['MemoryNode'] = field(default_factory=list)
    parent: Optional['MemoryNode'] = field(default=None, repr=False)

    @property
    def text(self) -> str:
        parts = [f"[{self.node_type}]"]
        for k, v in self.attributes.items():
            parts.append(f"{k}: {v}")
        return " | ".join(parts)

    @property
    def path(self) -> str:
        nodes = []
        cur = self
        while cur is not None:
            nodes.append(cur.node_type)
            cur = cur.parent
        return "/" + "/".join(reversed(nodes))

    def descendants(self) -> List['MemoryNode']:
        result = []
        queue = deque(self.children)
        while queue:
            node = queue.popleft()
            result.append(node)
            queue.extend(node.children)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], parent: Optional['MemoryNode'] = None) -> 'MemoryNode':
        node = cls(
            node_id=d["node_id"],
            node_type=d["node_type"],
            attributes=d.get("attributes", {}),
            parent=parent,
        )
        for child_dict in d.get("children", []):
            child = cls.from_dict(child_dict, parent=node)
            node.children.append(child)
        return node


class MemoryTree:
    """Rooted tree representation of structured memory."""

    def __init__(self, root: MemoryNode):
        self.root = root
        self._index: Dict[str, MemoryNode] = {}
        self._rebuild_index()

    def _rebuild_index(self):
        self._index.clear()
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            self._index[node.node_id] = node
            queue.extend(node.children)

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self._index.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[MemoryNode]:
        return [n for n in self._index.values() if n.node_type == node_type]

    def get_subtree_text(self, node: MemoryNode, depth: int = 0, max_depth: int = 10) -> str:
        if depth > max_depth:
            return ""
        indent = "  " * depth
        lines = [f"{indent}{node.text}"]
        for child in node.children:
            lines.append(self.get_subtree_text(child, depth + 1, max_depth))
        return "\n".join(lines)

    def to_text(self, max_depth: int = 10) -> str:
        return self.get_subtree_text(self.root, max_depth=max_depth)

    def to_flat_entries(self) -> List[str]:
        """Flatten tree into a list of text entries (for flat memory baseline)."""
        entries = []
        for node in self._index.values():
            if node.attributes:
                entries.append(node.text)
        return entries

    def keyword_search(self, query: str, topk: int = 5) -> List[Tuple[MemoryNode, float]]:
        """Simple keyword-based search over all nodes. Returns (node, score) pairs."""
        query_tokens = set(query.lower().split())
        scored = []
        for node in self._index.values():
            node_text = node.text.lower()
            node_tokens = set(node_text.split())
            if not query_tokens:
                continue
            overlap = len(query_tokens & node_tokens)
            score = overlap / len(query_tokens)
            if score > 0:
                scored.append((node, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

    def subtree_search(self, query: str, topk: int = 3) -> str:
        """Search for relevant nodes and return their subtrees as formatted text."""
        matches = self.keyword_search(query, topk=topk)
        if not matches:
            return "No relevant memory found."

        parts = []
        for node, score in matches:
            subtree_text = self.get_subtree_text(node, max_depth=3)
            path = node.path
            parts.append(f"Path: {path} (relevance: {score:.2f})\n{subtree_text}")
        return "\n---\n".join(parts)

    def navigate_path(self, path: str) -> Optional[MemoryNode]:
        """Navigate the tree using a slash-separated path of node types.
        E.g., '/Itinerary/Day/POI' returns the first matching node at each level.
        """
        parts = [p for p in path.strip("/").split("/") if p]
        current = self.root
        for part in parts:
            if current.node_type.lower() == part.lower():
                continue
            found = None
            for child in current.children:
                if child.node_type.lower() == part.lower():
                    found = child
                    break
            if found is None:
                return None
            current = found
        return current

    def semantic_navigate(self, query: str, topk: int = 3) -> str:
        """Navigate the tree using a natural-language query.

        Strategy: parse for node-type hints, then keyword-match within those subtrees.
        Falls back to global keyword search if no type hints are found.
        """
        known_types = {n.node_type.lower() for n in self._index.values()}
        query_lower = query.lower()

        target_type = None
        for t in known_types:
            if t in query_lower:
                target_type = t
                break

        if target_type:
            candidates = self.get_nodes_by_type(
                next(n.node_type for n in self._index.values() if n.node_type.lower() == target_type)
            )
            query_tokens = set(query_lower.split()) - {target_type}
            scored = []
            for node in candidates:
                node_text = node.text.lower()
                node_tokens = set(node_text.split())
                overlap = len(query_tokens & node_tokens) if query_tokens else 0
                score = overlap / max(len(query_tokens), 1)
                scored.append((node, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            matches = scored[:topk]
        else:
            matches = self.keyword_search(query, topk=topk)

        if not matches:
            return "No relevant memory found."

        parts = []
        for node, score in matches:
            subtree_text = self.get_subtree_text(node, max_depth=3)
            parts.append(f"Path: {node.path}\n{subtree_text}")
        return "\n---\n".join(parts)

    def to_json(self) -> str:
        return json.dumps(self.root.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'MemoryTree':
        data = json.loads(json_str)
        root = MemoryNode.from_dict(data)
        return cls(root)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryTree':
        root = MemoryNode.from_dict(data)
        return cls(root)

    def add_child(self, parent_id: str, child: MemoryNode) -> bool:
        parent = self.get_node(parent_id)
        if parent is None:
            return False
        child.parent = parent
        parent.children.append(child)
        self._index[child.node_id] = child
        for desc in child.descendants():
            self._index[desc.node_id] = desc
        return True

    def remove_node(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        if node is None or node is self.root:
            return False
        if node.parent:
            node.parent.children = [c for c in node.parent.children if c.node_id != node_id]
        for desc in node.descendants():
            self._index.pop(desc.node_id, None)
        self._index.pop(node_id, None)
        return True

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return f"MemoryTree(nodes={len(self)}, root={self.root.node_type})"
