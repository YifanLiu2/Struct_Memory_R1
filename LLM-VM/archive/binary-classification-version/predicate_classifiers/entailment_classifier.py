"""
Entailment-based predicate classifier.
Uses BART-NLI to score semantic relevance of nodes against predicates.
"""

import time
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_bart_client
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


class EntailmentPredicateClassifier(PredicateClassifier):
    """
    Entailment-based predicate classifier using BART-NLI.
    
    For leaf nodes: uses node's own description for entailment scoring.
    For parent nodes with subtree: averages scores of all children in subtree.
    """

    def __init__(self, threshold: float = None, hypothesis_template: str = None):
        """
        Initialize the classifier.
        
        Args:
            threshold: Entailment threshold (0-1). If None, loaded from config.
            hypothesis_template: Template for hypothesis. If None, loaded from config.
        """
        config = load_config().get("entailment", {})
        self.threshold = threshold if threshold is not None else config.get("threshold", 0.5)
        self.hypothesis_template = hypothesis_template or config.get(
            "hypothesis_template", 
            "This is related to {predicate}."
        )
        self._client = None  # Lazy load
    
    @property
    def client(self):
        """Lazy load the BART NLI client."""
        if self._client is None:
            self._client = get_bart_client()
        return self._client
    
    def classify_batch(
        self,
        nodes: list[NodeInfo],
        predicate: str
    ) -> BatchClassificationResult:
        """
        Classify multiple nodes against a predicate using entailment scoring.
        
        For nodes with subtree_text (parent nodes):
          - Extract children descriptions and score each
          - Use average score as the node's score
        
        For leaf nodes:
          - Score the node's own description
        
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
                classifier_type="entailment",
                duration_ms=0.0
            )
        
        results = []
        scores_detail = {
            "threshold": self.threshold,
            "hypothesis_template": self.hypothesis_template,
            "node_scores": []
        }
        
        for node in nodes:
            score, detail = self._score_node(node, predicate)
            is_match = score >= self.threshold
            
            reasoning = f"Entailment score: {score:.4f} (threshold: {self.threshold})"
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
            classifier_type="entailment",
            scores_detail=scores_detail,
            duration_ms=duration_ms
        )
    
    def _score_node(self, node: NodeInfo, predicate: str) -> tuple[float, dict]:
        """
        Score a single node against a predicate.
        
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
            text_parts.append(f"Name: {node.name}")
        if node.description:
            text_parts.append(f"Description: {node.description}")
        
        if not text_parts:
            # No description available - return 0
            detail["error"] = "No description available"
            return 0.0, detail
        
        node_text = "\n".join(text_parts)
        score = self.client.get_entailment_score(
            node_text, 
            predicate,
            hypothesis_template=self.hypothesis_template
        )
        
        detail["node_text"] = node_text[:200]  # Truncate for trace
        detail["direct_score"] = score
        
        return score, detail
    
    def _score_with_subtree(self, node: NodeInfo, predicate: str, detail: dict) -> tuple[float, dict]:
        """Score a parent node by averaging scores of its children."""
        detail["is_subtree_average"] = True
        
        # Parse children from subtree_text
        # Format: "  [1] Restaurant: Name\n      Description..."
        children_texts = self._parse_subtree_children(node)
        
        if not children_texts:
            # Fallback to node's own info if no children parsed
            detail["fallback"] = "No children parsed, using node info"
            return self._score_leaf(node, predicate, detail)
        
        # Score each child
        child_scores = self.client.batch_entailment_scores(
            children_texts,
            predicate,
            hypothesis_template=self.hypothesis_template
        )
        
        # Calculate average
        avg_score = sum(child_scores) / len(child_scores)
        
        detail["child_count"] = len(children_texts)
        detail["child_scores"] = [
            {"text": text[:100], "score": score}
            for text, score in zip(children_texts, child_scores)
        ]
        detail["average_score"] = avg_score
        
        return avg_score, detail
    
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
    
    classifier = EntailmentPredicateClassifier()
    
    print("Testing Entailment Predicate Classifier")
    print("=" * 60)
    print(f"Threshold: {classifier.threshold}")
    print(f"Hypothesis template: {classifier.hypothesis_template}")
    
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

