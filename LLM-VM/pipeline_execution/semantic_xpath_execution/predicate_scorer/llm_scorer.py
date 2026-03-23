"""
LLM-based predicate scorer.

Uses OpenAI API to score how well node descriptions match semantic predicates.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_default_client
from .base import PredicateScorer, ScoringResult, BatchScoringResult


logger = logging.getLogger(__name__)


class LLMPredicateScorer(PredicateScorer):
    """
    LLM-based implementation of predicate scoring.
    
    Sends batch requests to LLM to score multiple nodes against a predicate.
    """
    
    PROMPT_PATH = Path(__file__).parent.parent / "storage" / "prompts" / "predicate_scoring" / "predicate_scorer.txt"
    DEFAULT_TRACES_PATH = Path(__file__).parent.parent / "traces" / "reasoning_traces"
    
    def __init__(self, client=None, save_traces: bool = True, traces_path: Path = None):
        """
        Initialize the LLM scorer.
        
        Args:
            client: Optional OpenAI client. If not provided, one will be created.
            save_traces: Whether to save reasoning traces to disk.
            traces_path: Optional custom path for trace files. If None, traces are not saved.
        """
        self._client = client
        self._system_prompt = None
        self.save_traces = save_traces
        self.traces_path = traces_path
        
        # Only create traces directory if explicitly provided
        if self.traces_path:
            self.traces_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            self._client = get_default_client()
        return self._client
    
    @property
    def system_prompt(self) -> str:
        """Lazy load the system prompt from file."""
        if self._system_prompt is None:
            with open(self.PROMPT_PATH, "r", encoding="utf-8") as f:
                self._system_prompt = f.read()
        return self._system_prompt
    
    def score_batch(
        self, 
        nodes: List[Dict[str, Any]], 
        predicate: str
    ) -> BatchScoringResult:
        """
        Score multiple nodes against a predicate using LLM.
        
        Args:
            nodes: List of node dicts with 'id', 'type', 'description' keys
            predicate: The semantic predicate to match
            
        Returns:
            BatchScoringResult with scores for each node
        """
        if not nodes:
            return BatchScoringResult(predicate=predicate, results=[])
        
        # Build the user prompt with all nodes
        nodes_text = self._format_nodes_for_prompt(nodes)
        user_prompt = f"""Predicate: "{predicate}"

Nodes to score:
{nodes_text}

Score each node from 0.0 to 1.0 based on how well it matches the predicate.
Output JSON array with objects containing: id, score, reasoning"""
        
        # Call LLM
        try:
            result = self.client.complete_with_usage(
                user_prompt,
                system_prompt=self.system_prompt,
                temperature=0.1,
                max_tokens=2048
            )
            
            response = result.content
            
            # Parse response
            results = self._parse_response(response, nodes, predicate)
            
            # Save trace if enabled
            if self.save_traces:
                self._save_trace(predicate, nodes, response, results)
            
            return BatchScoringResult(
                predicate=predicate,
                results=results,
                metadata={"raw_response": response},
                token_usage=result.usage.to_dict()
            )
            
        except Exception as e:
            logger.error(f"Error scoring batch: {e}")
            # Return zero scores on error
            return BatchScoringResult(
                predicate=predicate,
                results=[
                    ScoringResult(
                        node_id=n.get("id", ""),
                        node_type=n.get("type", ""),
                        node_description=n.get("description", ""),
                        predicate=predicate,
                        score=0.0,
                        reasoning=f"Error: {e}"
                    )
                    for n in nodes
                ],
                metadata={"error": str(e)}
            )
    
    def _format_nodes_for_prompt(self, nodes: List[Dict[str, Any]]) -> str:
        """Format nodes for the prompt."""
        lines = []
        for node in nodes:
            node_id = node.get("id", "unknown")
            node_type = node.get("type", "unknown")
            description = node.get("description", "")
            name = node.get("name", "")
            
            # Include name if available for better context
            if name:
                lines.append(f"[{node_id}] ({node_type}) {name}: {description}")
            else:
                lines.append(f"[{node_id}] ({node_type}): {description}")
        
        return "\n".join(lines)
    
    def _parse_response(
        self, 
        response: str, 
        nodes: List[Dict[str, Any]], 
        predicate: str
    ) -> List[ScoringResult]:
        """Parse LLM response into ScoringResult objects."""
        results = []
        
        # Try to extract JSON from response
        try:
            # Find JSON array in response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Create a map for quick lookup
                node_map = {n.get("id"): n for n in nodes}
                
                for item in parsed:
                    node_id = str(item.get("id", ""))
                    score = float(item.get("score", 0.0))
                    reasoning = item.get("reasoning", "")
                    
                    # Clamp score to [0, 1]
                    score = max(0.0, min(1.0, score))
                    
                    node = node_map.get(node_id, {})
                    results.append(ScoringResult(
                        node_id=node_id,
                        node_type=node.get("type", ""),
                        node_description=node.get("description", ""),
                        predicate=predicate,
                        score=score,
                        reasoning=reasoning
                    ))
                
                return results
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
        
        # Fallback: return zero scores for all nodes
        for node in nodes:
            results.append(ScoringResult(
                node_id=node.get("id", ""),
                node_type=node.get("type", ""),
                node_description=node.get("description", ""),
                predicate=predicate,
                score=0.0,
                reasoning="Failed to parse LLM response"
            ))
        
        return results
    
    def _save_trace(
        self, 
        predicate: str, 
        nodes: List[Dict[str, Any]], 
        response: str,
        results: List[ScoringResult]
    ):
        """Save reasoning trace to disk."""
        if not self.save_traces or self.traces_path is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_file = self.traces_path / f"scoring_{timestamp}.json"
        
        trace_data = {
            "timestamp": timestamp,
            "predicate": predicate,
            "nodes": nodes,
            "raw_response": response,
            "results": [
                {
                    "node_id": r.node_id,
                    "node_type": r.node_type,
                    "score": r.score,
                    "reasoning": r.reasoning
                }
                for r in results
            ]
        }
        
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)
        
        logger.debug(f"Saved scoring trace to {trace_file}")

