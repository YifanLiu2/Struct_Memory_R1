"""
Base class for predicate classifiers.
Supports batch classification of nodes against semantic predicates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from xml.etree.ElementTree import Element


@dataclass
class NodeInfo:
    """Information about a node for classification."""
    element: Element  # The XML element
    tree_path: str  # Path in tree, e.g., "/Itinerary/Day[1]/Restaurant[2]"
    node_type: str  # e.g., "Restaurant", "POI", "Day"
    name: Optional[str] = None
    description: Optional[str] = None
    subtree_text: Optional[str] = None  # Text representation of children
    
    @classmethod
    def from_element(cls, element: Element, tree_path: str) -> "NodeInfo":
        """Create NodeInfo from an XML element."""
        node_type = element.tag
        name = None
        description = None
        
        # Extract name and description from child elements
        name_elem = element.find("name")
        if name_elem is not None and name_elem.text:
            name = name_elem.text.strip()
        
        desc_elem = element.find("description")
        if desc_elem is not None and desc_elem.text:
            description = desc_elem.text.strip()
        
        return cls(
            element=element,
            tree_path=tree_path,
            node_type=node_type,
            name=name,
            description=description
        )
    
    def to_text(self) -> str:
        """Convert node info to text for classification."""
        parts = [f"Type: {self.node_type}"]
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.subtree_text:
            parts.append(f"Children:\n{self.subtree_text}")
        return "\n".join(parts)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tree_path": self.tree_path,
            "node_type": self.node_type,
            "name": self.name,
            "description": self.description[:100] if self.description else None
        }


@dataclass
class ClassificationResult:
    """Result of classifying a node against a predicate."""
    node_info: NodeInfo
    predicate: str
    is_match: bool
    confidence: float = 1.0  # 0.0 to 1.0
    reasoning: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "tree_path": self.node_info.tree_path,
            "name": self.node_info.name,
            "is_match": self.is_match,
            "confidence": self.confidence
        }
        if self.reasoning:
            result["reasoning"] = self.reasoning
        return result


@dataclass
class BatchClassificationResult:
    """Result of batch classification with full details for tracing."""
    predicate: str
    input_nodes: list[NodeInfo]
    results: list[ClassificationResult]
    classifier_type: str = "llm"  # "llm", "entailment", or "cosine"
    prompt_sent: Optional[str] = None  # For LLM classifier
    raw_response: Optional[str] = None  # For LLM classifier
    scores_detail: Optional[dict] = None  # For entailment/cosine classifier
    duration_ms: Optional[float] = None  # Time taken for this batch classification
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "classifier_type": self.classifier_type,
            "predicate": self.predicate,
            "input_count": len(self.input_nodes),
            "input_nodes": [n.to_dict() for n in self.input_nodes],
            "results": [r.to_dict() for r in self.results],
            "matched_count": sum(1 for r in self.results if r.is_match),
            "matched_paths": [r.node_info.tree_path for r in self.results if r.is_match],
        }
        if self.duration_ms is not None:
            result["duration_ms"] = round(self.duration_ms, 2)
        if self.prompt_sent:
            result["prompt_sent"] = self.prompt_sent
        if self.raw_response:
            result["raw_response"] = self.raw_response
        if self.scores_detail:
            result["scores_detail"] = self.scores_detail
        return result


class PredicateClassifier(ABC):
    """
    Abstract base class for predicate classifiers.
    Supports batch classification for efficiency.
    """
    
    @abstractmethod
    def classify_batch(
        self,
        nodes: list[NodeInfo],
        predicate: str
    ) -> BatchClassificationResult:
        """
        Classify multiple nodes against a single predicate.
        
        Args:
            nodes: List of NodeInfo objects to classify
            predicate: The semantic predicate (e.g., "italian", "jazz")
            
        Returns:
            BatchClassificationResult with full details
        """
        pass
    
    def classify_single(
        self,
        node: NodeInfo,
        predicate: str
    ) -> ClassificationResult:
        """
        Classify a single node against a predicate.
        Default implementation calls classify_batch with single item.
        
        Args:
            node: NodeInfo to classify
            predicate: The semantic predicate
            
        Returns:
            ClassificationResult
        """
        batch_result = self.classify_batch([node], predicate)
        return batch_result.results[0]
    
    def get_matching_nodes(
        self,
        nodes: list[NodeInfo],
        predicate: str
    ) -> tuple[list[NodeInfo], BatchClassificationResult]:
        """
        Filter nodes to only those matching the predicate.
        
        Args:
            nodes: List of NodeInfo objects
            predicate: The semantic predicate
            
        Returns:
            Tuple of (matching nodes, batch result for tracing)
        """
        batch_result = self.classify_batch(nodes, predicate)
        matching = [r.node_info for r in batch_result.results if r.is_match]
        return matching, batch_result
