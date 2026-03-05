"""
Flat memory store for Memory-R1 baseline.

Implements a simple list-based memory bank with embedding-based retrieval,
mirroring the flat memory representation used in Memory-R1 (Yan et al., 2025).
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    entry_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class FlatMemoryStore:
    """List-based memory bank with cosine-similarity retrieval."""

    def __init__(self, embedding_fn=None):
        self.entries: List[MemoryEntry] = []
        self._embedding_fn = embedding_fn
        self._id_counter = 0

    def add(self, text: str, metadata: Optional[Dict[str, str]] = None) -> MemoryEntry:
        entry = MemoryEntry(
            entry_id=str(self._id_counter),
            text=text,
            metadata=metadata or {},
        )
        if self._embedding_fn is not None:
            entry.embedding = self._embedding_fn(text)
        self._id_counter += 1
        self.entries.append(entry)
        return entry

    def add_batch(self, texts: List[str], metadata_list: Optional[List[Dict[str, str]]] = None):
        metadata_list = metadata_list or [{}] * len(texts)
        for text, meta in zip(texts, metadata_list):
            self.add(text, meta)

    def retrieve(self, query: str, topk: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """Retrieve top-k entries by cosine similarity."""
        if not self.entries:
            return []

        if self._embedding_fn is not None:
            return self._retrieve_by_embedding(query, topk)
        return self._retrieve_by_keyword(query, topk)

    def _retrieve_by_embedding(self, query: str, topk: int) -> List[Tuple[MemoryEntry, float]]:
        query_emb = self._embedding_fn(query)
        scores = []
        for entry in self.entries:
            if entry.embedding is None:
                entry.embedding = self._embedding_fn(entry.text)
            score = self._cosine_similarity(query_emb, entry.embedding)
            scores.append((entry, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

    def _retrieve_by_keyword(self, query: str, topk: int) -> List[Tuple[MemoryEntry, float]]:
        """Fallback keyword-based retrieval when no embedding function is available."""
        query_tokens = set(query.lower().split())
        scores = []
        for entry in self.entries:
            entry_tokens = set(entry.text.lower().split())
            if not query_tokens:
                continue
            overlap = len(query_tokens & entry_tokens)
            score = overlap / len(query_tokens)
            scores.append((entry, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def format_results(self, results: List[Tuple[MemoryEntry, float]]) -> str:
        """Format retrieval results as text for injection into LLM context."""
        if not results:
            return "No relevant memories found."
        parts = []
        for idx, (entry, score) in enumerate(results):
            meta_str = ""
            if entry.metadata:
                meta_str = " | ".join(f"{k}: {v}" for k, v in entry.metadata.items())
                meta_str = f" ({meta_str})"
            parts.append(f"Memory {idx+1}{meta_str}: {entry.text}")
        return "\n".join(parts)

    def update(self, entry_id: str, new_text: str) -> bool:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                entry.text = new_text
                if self._embedding_fn is not None:
                    entry.embedding = self._embedding_fn(new_text)
                return True
        return False

    def delete(self, entry_id: str) -> bool:
        for i, entry in enumerate(self.entries):
            if entry.entry_id == entry_id:
                self.entries.pop(i)
                return True
        return False

    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {"entry_id": e.entry_id, "text": e.text, "metadata": e.metadata}
            for e in self.entries
        ]

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]], embedding_fn=None) -> 'FlatMemoryStore':
        store = cls(embedding_fn=embedding_fn)
        for item in data:
            entry = MemoryEntry(
                entry_id=item["entry_id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            store.entries.append(entry)
            store._id_counter = max(store._id_counter, int(item["entry_id"]) + 1)
        return store

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str, embedding_fn=None) -> 'FlatMemoryStore':
        data = json.loads(json_str)
        return cls.from_dict(data, embedding_fn=embedding_fn)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"FlatMemoryStore(entries={len(self)})"
