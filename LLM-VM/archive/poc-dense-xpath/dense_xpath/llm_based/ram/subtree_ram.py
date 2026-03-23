"""
Subtree RAM - Stores subtree information for parent nodes during query execution.

For queries like "relaxing day with italian restaurant":
1. First evaluate children (find "italian restaurant")
2. Store matching children in RAM keyed by parent's tree_path
3. When evaluating parent predicate ("relaxing"), retrieve stored subtree
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime


@dataclass
class SubtreeEntry:
    """Entry storing a node's matching subtree for a predicate."""
    tree_path: str                    # Parent node's tree path
    node_type: str                    # Parent node's type
    node_name: Optional[str]          # Parent node's name
    predicate: str                    # The predicate being evaluated
    matched_children: list[dict]      # Children that matched downstream predicates
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "SubtreeEntry":
        return cls(**d)


class SubtreeRAM:
    """
    In-memory storage for subtree information during query execution.
    Persists to disk for debugging/inspection.
    """
    
    RAM_DIR = Path(__file__).parent
    
    def __init__(self, query_id: str = None):
        """
        Initialize the RAM storage.
        
        Args:
            query_id: Optional identifier for this query execution
        """
        self.query_id = query_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._storage: dict[str, SubtreeEntry] = {}  # tree_path -> SubtreeEntry
    
    def _make_key(self, tree_path: str, predicate: str) -> str:
        """Create storage key from tree path and predicate."""
        return f"{tree_path}|{predicate}"
    
    def store(
        self,
        tree_path: str,
        node: dict,
        predicate: str,
        matched_children: list[dict]
    ) -> None:
        """
        Store subtree information for a parent node.
        
        Args:
            tree_path: The parent node's tree path
            node: The parent node dict
            predicate: The predicate being evaluated on this node
            matched_children: Children that matched downstream predicates
        """
        key = self._make_key(tree_path, predicate)
        entry = SubtreeEntry(
            tree_path=tree_path,
            node_type=node.get("type", "unknown"),
            node_name=node.get("name"),
            predicate=predicate,
            matched_children=matched_children
        )
        self._storage[key] = entry
    
    def retrieve(self, tree_path: str, predicate: str) -> Optional[list[dict]]:
        """
        Retrieve stored subtree information.
        
        Args:
            tree_path: The parent node's tree path
            predicate: The predicate being evaluated
            
        Returns:
            List of matched children, or None if not found
        """
        key = self._make_key(tree_path, predicate)
        entry = self._storage.get(key)
        if entry:
            return entry.matched_children
        return None
    
    def get_entry(self, tree_path: str, predicate: str) -> Optional[SubtreeEntry]:
        """Get full entry for a node/predicate combination."""
        key = self._make_key(tree_path, predicate)
        return self._storage.get(key)
    
    def get_all_entries(self) -> list[SubtreeEntry]:
        """Get all stored entries."""
        return list(self._storage.values())
    
    def clear(self) -> None:
        """Clear all stored data."""
        self._storage.clear()
    
    def save_to_disk(self, filename: str = None) -> Path:
        """
        Save current RAM state to disk for debugging.
        
        Args:
            filename: Optional filename (defaults to query_id)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"ram_{self.query_id}.json"
        
        filepath = self.RAM_DIR / filename
        
        data = {
            "query_id": self.query_id,
            "timestamp": datetime.now().isoformat(),
            "entries_count": len(self._storage),
            "entries": [entry.to_dict() for entry in self._storage.values()]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def __len__(self) -> int:
        return len(self._storage)
    
    def __repr__(self) -> str:
        return f"SubtreeRAM(query_id={self.query_id}, entries={len(self._storage)})"


if __name__ == "__main__":
    # Quick test
    ram = SubtreeRAM("test_query")
    
    # Simulate storing subtree for Day 1
    parent_node = {
        "type": "Day",
        "name": "Day 1",
        "description": "First day of Toronto trip"
    }
    
    matched_children = [
        {
            "type": "Restaurant",
            "name": "Buca Yorkville",
            "description": "Upscale Italian dining"
        }
    ]
    
    ram.store(
        tree_path="/Itinerary[1]/Day[1]",
        node=parent_node,
        predicate="italian",
        matched_children=matched_children
    )
    
    print(f"RAM: {ram}")
    print(f"Entries: {ram.get_all_entries()}")
    
    # Retrieve
    children = ram.retrieve("/Itinerary[1]/Day[1]", "italian")
    print(f"Retrieved children: {children}")
    
    # Save to disk
    filepath = ram.save_to_disk()
    print(f"Saved to: {filepath}")

