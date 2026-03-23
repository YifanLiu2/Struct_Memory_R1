"""
Base class for predicate scorers.

This module defines the abstract interface for scoring nodes against semantic predicates.
Different implementations can use LLM, entailment models, cosine similarity, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ScoringResult:
    """Result of scoring a single node against a predicate."""
    node_id: str
    node_type: str
    node_description: str
    predicate: str
    score: float  # 0.0 to 1.0
    reasoning: str = ""


@dataclass 
class BatchScoringResult:
    """Result of batch scoring multiple nodes against a predicate."""
    predicate: str
    results: List[ScoringResult]
    metadata: Dict[str, Any] = None
    token_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PredicateScorer(ABC):
    """
    Abstract base class for predicate scorers.
    
    Implementations should score how well a node's description matches a semantic predicate.
    """
    
    @abstractmethod
    def score_batch(
        self, 
        nodes: List[Dict[str, Any]], 
        predicate: str
    ) -> BatchScoringResult:
        """
        Score multiple nodes against a single predicate in one batch call.
        
        Args:
            nodes: List of node dicts with 'id', 'type', 'description' keys
            predicate: The semantic predicate to match (e.g., "artistic", "italian")
            
        Returns:
            BatchScoringResult with scores for each node
        """
        pass
    
    def score_single(
        self, 
        node: Dict[str, Any], 
        predicate: str
    ) -> ScoringResult:
        """
        Score a single node against a predicate.
        Default implementation uses batch scoring with a single node.
        
        Args:
            node: Node dict with 'id', 'type', 'description' keys
            predicate: The semantic predicate to match
            
        Returns:
            ScoringResult for the node
        """
        batch_result = self.score_batch([node], predicate)
        return batch_result.results[0] if batch_result.results else None

