"""
Memory retrieval server for Structured Memory-R1.

FastAPI server analogous to search_r1/search/retrieval_server.py, but
serves from flat or structured (tree) memory stores instead of a corpus index.
Exposes a POST /retrieve endpoint with the same interface as the search server.
"""

import json
import os
import argparse
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from memory_r1.memory_tree import MemoryTree
from memory_r1.flat_memory import FlatMemoryStore


class MemoryQueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 5
    return_scores: bool = True
    memory_type: Optional[str] = "flat"  # "flat" or "structured"


app = FastAPI(title="Memory-R1 Retrieval Server")

flat_store: Optional[FlatMemoryStore] = None
tree_store: Optional[MemoryTree] = None


def load_flat_memory(path: str) -> FlatMemoryStore:
    with open(path, "r") as f:
        data = json.load(f)
    store = FlatMemoryStore()
    for item in data:
        store.add(
            text=item["text"],
            metadata=item.get("metadata", {}),
        )
    return store


def load_structured_memory(path: str) -> MemoryTree:
    with open(path, "r") as f:
        data = json.load(f)
    return MemoryTree.from_dict(data)


def flat_retrieve(queries: List[str], topk: int, return_scores: bool) -> List[List[Dict]]:
    results = []
    for query in queries:
        hits = flat_store.retrieve(query, topk=topk)
        query_results = []
        for entry, score in hits:
            doc = {
                "contents": entry.text,
                "entry_id": entry.entry_id,
                "metadata": entry.metadata,
            }
            if return_scores:
                query_results.append({"document": doc, "score": score})
            else:
                query_results.append(doc)
        results.append(query_results)
    return results


def structured_retrieve(queries: List[str], topk: int, return_scores: bool) -> List[List[Dict]]:
    results = []
    for query in queries:
        matches = tree_store.keyword_search(query, topk=topk)
        query_results = []
        for node, score in matches:
            subtree_text = tree_store.get_subtree_text(node, max_depth=3)
            doc = {
                "contents": subtree_text,
                "node_id": node.node_id,
                "node_type": node.node_type,
                "path": node.path,
            }
            if return_scores:
                query_results.append({"document": doc, "score": score})
            else:
                query_results.append(doc)
        results.append(query_results)
    return results


@app.post("/retrieve")
def retrieve_endpoint(request: MemoryQueryRequest):
    """
    Endpoint matching the Search-R1 retrieval API interface.

    Input:
    {
      "queries": ["What is the user's dietary preference?"],
      "topk": 3,
      "return_scores": true,
      "memory_type": "flat" | "structured"
    }
    """
    topk = request.topk or 5
    memory_type = request.memory_type or "flat"

    if memory_type == "structured" and tree_store is not None:
        result = structured_retrieve(request.queries, topk, request.return_scores)
    elif flat_store is not None:
        result = flat_retrieve(request.queries, topk, request.return_scores)
    else:
        result = [[] for _ in request.queries]

    return {"result": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the memory retrieval server.")
    parser.add_argument(
        "--flat_memory_path", type=str, default=None,
        help="Path to flat memory JSON file.",
    )
    parser.add_argument(
        "--structured_memory_path", type=str, default=None,
        help="Path to structured memory tree JSON file.",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    if args.flat_memory_path:
        flat_store = load_flat_memory(args.flat_memory_path)
        print(f"Loaded flat memory with {len(flat_store)} entries")

    if args.structured_memory_path:
        tree_store = load_structured_memory(args.structured_memory_path)
        print(f"Loaded structured memory with {len(tree_store)} nodes")

    uvicorn.run(app, host=args.host, port=args.port)
