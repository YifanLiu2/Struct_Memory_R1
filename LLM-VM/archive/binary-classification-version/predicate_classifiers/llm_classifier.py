"""
LLM-based predicate classifier.
Uses LLM to batch classify nodes against semantic predicates.
"""

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_client
from .base import (
    PredicateClassifier, 
    NodeInfo, 
    ClassificationResult, 
    BatchClassificationResult
)


class LLMPredicateClassifier(PredicateClassifier):
    """
    LLM-based predicate classifier that processes nodes in batches.
    """
    
    PROMPT_PATH = Path(__file__).parent.parent / "storage" / "prompts" / "predicate_scoring" / "predicate_scorer.txt"

    def __init__(self, client=None):
        """
        Initialize the classifier.
        
        Args:
            client: Optional OpenAI client. If not provided, one will be created.
        """
        self._client = client
        self._system_prompt = None  # Lazy load
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            self._client = get_client()
        return self._client
    
    @property
    def system_prompt(self) -> str:
        """Lazy load the system prompt from file."""
        if self._system_prompt is None:
            with open(self.PROMPT_PATH, "r") as f:
                self._system_prompt = f.read()
        return self._system_prompt
    
    def classify_batch(
        self,
        nodes: list[NodeInfo],
        predicate: str
    ) -> BatchClassificationResult:
        """
        Classify multiple nodes against a predicate using LLM.
        
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
                classifier_type="llm",
                prompt_sent="",
                raw_response="",
                duration_ms=0.0
            )
        
        # Build prompt with all nodes
        prompt = self._build_batch_prompt(nodes, predicate)
        
        # Call LLM (increased max_tokens for reasoning output)
        response = self.client.complete(
            prompt,
            system_prompt=self.system_prompt,
            temperature=0.0,
            max_tokens=2048
        )
        
        # Parse response
        parsed_results = self._parse_response(response, len(nodes))
        
        # Build results
        results = []
        for i, node in enumerate(nodes):
            parsed = parsed_results.get(i, {"match": False, "reasoning": None})
            is_match = parsed.get("match", False)
            reasoning = parsed.get("reasoning")
            results.append(ClassificationResult(
                node_info=node,
                predicate=predicate,
                is_match=is_match,
                confidence=1.0 if is_match else 0.0,
                reasoning=reasoning
            ))
        
        duration_ms = (time.time() - start_time) * 1000
        
        return BatchClassificationResult(
            predicate=predicate,
            input_nodes=nodes,
            results=results,
            classifier_type="llm",
            prompt_sent=prompt,
            raw_response=response,
            duration_ms=duration_ms
        )
    
    def _build_batch_prompt(self, nodes: list[NodeInfo], predicate: str) -> str:
        """Build the prompt for batch classification."""
        lines = [f"Predicate: \"{predicate}\"", "", "Nodes:"]
        
        for i, node in enumerate(nodes):
            lines.append(f"\n[{i}] {node.tree_path}")
            lines.append(node.to_text())
        
        lines.append("\nClassify each node. Output JSON only.")
        
        return "\n".join(lines)
    
    def _parse_response(self, response: str, expected_count: int) -> dict[int, dict]:
        """
        Parse LLM response to extract match results and reasoning.
        
        Returns:
            Dict mapping node index to {"match": bool, "reasoning": str}
        """
        parsed = {}
        
        try:
            # Try to extract JSON from response
            response = response.strip()
            
            # Handle case where LLM wraps in markdown
            if response.startswith("```"):
                lines = response.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```") and not in_json:
                        in_json = True
                        continue
                    elif line.startswith("```") and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                response = "\n".join(json_lines)
            
            data = json.loads(response)
            
            if "results" in data:
                for item in data["results"]:
                    idx = item.get("index", -1)
                    if 0 <= idx < expected_count:
                        parsed[idx] = {
                            "match": item.get("match", False),
                            "reasoning": item.get("reasoning")
                        }
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # If parsing fails, default to no matches
            pass
        
        return parsed


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
    
    classifier = LLMPredicateClassifier()
    
    print("Testing LLM Predicate Classifier")
    print("=" * 60)
    
    # Test with "italian" predicate
    predicate = "italian"
    print(f"\nPredicate: {predicate}")
    batch_result = classifier.classify_batch(nodes, predicate)
    
    print(f"\nPrompt sent:\n{batch_result.prompt_sent[:500]}...")
    print(f"\nRaw response:\n{batch_result.raw_response}")
    print(f"\nResults:")
    for r in batch_result.results:
        status = "✓ MATCH" if r.is_match else "✗ NO MATCH"
        print(f"  {status}: {r.node_info.name} ({r.node_info.tree_path})")
