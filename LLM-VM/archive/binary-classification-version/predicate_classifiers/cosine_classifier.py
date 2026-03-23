"""
Cosine similarity-based predicate classifier.
Uses TAS-B embeddings to compute semantic similarity between node content and predicates.
"""

import time
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_tas_b_client
from .base import (
    PredicateClassifier, 
    NodeInfo, 
    ClassificationResult, 
    BatchClassificationResult
)


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class CosinePredicateClassifier(PredicateClassifier):
    """
    Cosine similarity-based predicate classifier using TAS-B embeddings.
    
    For leaf nodes: computes cosine similarity between node description and predicate.
    For parent nodes with subtree: averages similarity scores of all children in subtree.
    """

    def __init__(self, threshold: float = None):
        """
        Initialize the classifier.
        
        Args:
            threshold: Similarity threshold (0-1). If None, loaded from config.
        """
        config = load_config().get("cosine_similarity", {})
        self.threshold = threshold if threshold is not None else config.get("threshold", 0.5)
        self._client = None  # Lazy load
    
    @property
    def client(self):
        """Lazy load the TAS-B client."""
        if self._client is None:
            self._client = get_tas_b_client()
        return self._client
    
    def classify_batch(
        self,
        nodes: list[NodeInfo],
        predicate: str
    ) -> BatchClassificationResult:
        """
        Classify multiple nodes against a predicate using cosine similarity.
        
        For nodes with subtree_text (parent nodes):
          - Extract children descriptions and compute similarity for each
          - Use average similarity as the node's score
        
        For leaf nodes:
          - Compute similarity with the node's own description
        
        Args:
            nodes: List of NodeInfo objects to classify
            predicate: The semantic predicate (e.g., "italian", "jazz")
            
        Returns:
            BatchClassificationResult with full details for tracing
        """
        start_time = time.time()
        
        if not nodes:
            return BatchClassificationResult(
                predicate=predicate,
                input_nodes=[],
                results=[],
                classifier_type="cosine",
                duration_ms=0.0
            )
        
        results = []
        scores_detail = {
            "threshold": self.threshold,
            "node_scores": []
        }
        
        for node in nodes:
            score, detail = self._score_node(node, predicate)
            is_match = score >= self.threshold
            
            reasoning = f"Cosine similarity: {score:.4f} (threshold: {self.threshold})"
            if detail.get("is_subtree_average"):
                reasoning += f" - averaged from {detail['child_count']} children"
            
            results.append(ClassificationResult(
                node_info=node,
                predicate=predicate,
                is_match=is_match,
                confidence=score,
                reasoning=reasoning
            ))
            
            scores_detail["node_scores"].append({
                "tree_path": node.tree_path,
                "score": score,
                "is_match": is_match,
                **detail
            })
        
        duration_ms = (time.time() - start_time) * 1000
        
        return BatchClassificationResult(
            predicate=predicate,
            input_nodes=nodes,
            results=results,
            classifier_type="cosine",
            scores_detail=scores_detail,
            duration_ms=duration_ms
        )
    
    def _score_node(self, node: NodeInfo, predicate: str) -> tuple[float, dict]:
        """
        Score a single node against a predicate using cosine similarity.
        
        Returns:
            Tuple of (score, detail_dict)
        """
        detail = {}
        
        # If node has subtree_text, it's a parent node - score children
        if node.subtree_text:
            return self._score_with_subtree(node, predicate, detail)
        
        # Leaf node - score own description
        return self._score_leaf(node, predicate, detail)
    
    def _score_leaf(self, node: NodeInfo, predicate: str, detail: dict) -> tuple[float, dict]:
        """Score a leaf node using its own description."""
        detail["is_subtree_average"] = False
        
        # Build text from node info
        text_parts = []
        if node.name:
            text_parts.append(node.name)
        if node.description:
            text_parts.append(node.description)
        
        if not text_parts:
            # No description available - return 0
            detail["error"] = "No description available"
            return 0.0, detail
        
        node_text = " - ".join(text_parts)
        score = self.client.similarity(node_text, predicate)
        
        detail["node_text"] = node_text[:200]  # Truncate for trace
        detail["direct_score"] = score
        
        return score, detail
    
    def _score_with_subtree(self, node: NodeInfo, predicate: str, detail: dict) -> tuple[float, dict]:
        """Score a parent node by averaging similarity scores of its children."""
        detail["is_subtree_average"] = True
        
        # Parse children from subtree_text
        children_texts = self._parse_subtree_children(node)
        
        if not children_texts:
            # Fallback to node's own info if no children parsed
            detail["fallback"] = "No children parsed, using node info"
            return self._score_leaf(node, predicate, detail)
        
        # Score each child using batch embeddings for efficiency
        child_scores = self._batch_similarity(children_texts, predicate)
        
        # Calculate average
        avg_score = sum(child_scores) / len(child_scores)
        
        detail["child_count"] = len(children_texts)
        detail["child_scores"] = [
            {"text": text[:100], "score": score}
            for text, score in zip(children_texts, child_scores)
        ]
        detail["average_score"] = avg_score
        
        return avg_score, detail
    
    def _batch_similarity(self, texts: list[str], predicate: str) -> list[float]:
        """
        Compute batch cosine similarity between texts and a predicate.
        """
        # Get predicate embedding once
        predicate_emb = self.client.get_embedding(predicate, normalize=True)
        
        # Get embeddings for all texts
        text_embeddings = self.client.get_embeddings(texts, normalize=True)
        
        # Compute dot products (cosine similarity since normalized)
        import numpy as np
        scores = np.dot(text_embeddings, predicate_emb).tolist()
        
        return scores
    
    def _parse_subtree_children(self, node: NodeInfo) -> list[str]:
        """
        Parse children descriptions from subtree_text.
        
        Format expected:
          [1] Type: Name
              Description text
          [2] Type: Name
              Description text
        """
        if not node.subtree_text:
            return []
        
        children = []
        current_child = []
        
        for line in node.subtree_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new child entry (starts with [n])
            if line.startswith("[") and "]" in line:
                # Save previous child if exists
                if current_child:
                    children.append(" ".join(current_child))
                # Start new child
                # Extract the part after [n]
                idx = line.index("]")
                current_child = [line[idx+1:].strip()]
            else:
                # Continue current child (description line)
                if current_child:
                    current_child.append(line)
        
        # Don't forget last child
        if current_child:
            children.append(" ".join(current_child))
        
        return children


if __name__ == "__main__":
    # Quick test
    from xml.etree.ElementTree import Element, SubElement
    
    # Create test nodes
    nodes = []
    
    # Node 1: Italian restaurant
    elem1 = Element("Restaurant")
    SubElement(elem1, "name").text = "Buca Yorkville"
    SubElement(elem1, "description").text = "Upscale Italian dining featuring traditional dishes with a modern twist."
    nodes.append(NodeInfo.from_element(elem1, "/Itinerary/Day[1]/Restaurant[1]"))
    
    # Node 2: Jazz bar
    elem2 = Element("Restaurant")
    SubElement(elem2, "name").text = "The Rex Hotel Jazz & Blues Bar"
    SubElement(elem2, "description").text = "Live jazz performances in a cozy, iconic Toronto venue."
    nodes.append(NodeInfo.from_element(elem2, "/Itinerary/Day[1]/Restaurant[2]"))
    
    # Node 3: French bistro
    elem3 = Element("Restaurant")
    SubElement(elem3, "name").text = "Cluny Bistro"
    SubElement(elem3, "description").text = "French-inspired bistro serving classic and modern dishes."
    nodes.append(NodeInfo.from_element(elem3, "/Itinerary/Day[3]/Restaurant[1]"))
    
    classifier = CosinePredicateClassifier()
    
    print("Testing Cosine Similarity Predicate Classifier")
    print("=" * 60)
    print(f"Threshold: {classifier.threshold}")
    
    # Test with "italian" predicate
    predicate = "italian"
    print(f"\nPredicate: {predicate}")
    batch_result = classifier.classify_batch(nodes, predicate)
    
    print(f"\nResults:")
    for r in batch_result.results:
        status = "✓ MATCH" if r.is_match else "✗ NO MATCH"
        print(f"  {status}: {r.node_info.name}")
        print(f"    Score: {r.confidence:.4f}")
        print(f"    {r.reasoning}")
    
    # Print scores detail
    print(f"\nScores Detail:")
    import json
    print(json.dumps(batch_result.scores_detail, indent=2, default=str))

