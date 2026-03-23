"""
Entailment-based predicate scorer.

Uses BART NLI model to score how well node descriptions match semantic predicates.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_bart_client, BartNLIClient
from .base import PredicateScorer, ScoringResult, BatchScoringResult


logger = logging.getLogger(__name__)


class EntailmentPredicateScorer(PredicateScorer):
    """
    Entailment-based implementation of predicate scoring.
    
    Uses BART Large MNLI model to calculate entailment scores between
    node descriptions and semantic predicates.
    """
    
    DEFAULT_TRACES_PATH = Path(__file__).parent.parent / "traces" / "reasoning_traces"
    
    def __init__(
        self, 
        client: BartNLIClient = None, 
        model_name: str = "facebook/bart-large-mnli",
        hypothesis_template: str = "This example is {predicate}.",
        save_traces: bool = True,
        traces_path: Path = None
    ):
        """
        Initialize the entailment scorer.
        
        Args:
            client: Optional BART NLI client. If not provided, one will be created.
            hypothesis_template: Template for constructing hypothesis from predicate.
                                 Use {predicate} as placeholder.
            save_traces: Whether to save reasoning traces to disk.
            traces_path: Optional custom path for trace files. If None, traces are not saved.
        """
        self._client = client
        self.model_name = model_name
        self.hypothesis_template = hypothesis_template
        self.save_traces = save_traces
        self.traces_path = traces_path
        
        # Only create traces directory if explicitly provided
        if self.traces_path:
            self.traces_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def client(self) -> BartNLIClient:
        """Lazy load the BART NLI client."""
        if self._client is None:
            self._client = get_bart_client()
        return self._client
    
    def score_batch(
        self, 
        nodes: List[Dict[str, Any]], 
        predicate: str
    ) -> BatchScoringResult:
        """
        Score multiple nodes against a predicate using entailment.
        
        Args:
            nodes: List of node dicts with 'id', 'type', 'description' keys
            predicate: The semantic predicate to match
            
        Returns:
            BatchScoringResult with scores for each node
        """
        if not nodes:
            return BatchScoringResult(predicate=predicate, results=[])
        
        # Build node info strings for entailment
        node_infos = []
        for node in nodes:
            node_type = node.get("type", "")
            name = node.get("name", "")
            description = node.get("description", "")
            
            # Construct informative premise for NLI
            if name and description:
                info = f"{node_type}: {name} - {description}"
            elif description:
                info = f"{node_type}: {description}"
            elif name:
                info = f"{node_type}: {name}"
            else:
                info = f"{node_type}"
            
            node_infos.append(info)
        
        try:
            # Batch score using entailment
            scores = self.client.batch_entailment_scores(
                node_infos, 
                predicate,
                hypothesis_template=self.hypothesis_template
            )
            
            # Build results
            results = []
            for i, node in enumerate(nodes):
                score = scores[i] if i < len(scores) else 0.0
                results.append(ScoringResult(
                    node_id=node.get("id", f"node_{i}"),
                    node_type=node.get("type", ""),
                    node_description=node.get("description", ""),
                    predicate=predicate,
                    score=score,
                    reasoning=f"Entailment score: {score:.4f}"
                ))
            
            # Save trace if enabled
            if self.save_traces:
                self._save_trace(predicate, nodes, node_infos, scores, results)
            
            return BatchScoringResult(
                predicate=predicate,
                results=results,
                metadata={
                    "method": "entailment",
                    "hypothesis_template": self.hypothesis_template,
                    "model": self.client.model_name
                }
            )
            
        except Exception as e:
            logger.error(f"Error scoring batch with entailment: {e}")
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
    
    def _save_trace(
        self, 
        predicate: str, 
        nodes: List[Dict[str, Any]], 
        node_infos: List[str],
        scores: List[float],
        results: List[ScoringResult]
    ):
        """Save reasoning trace to disk."""
        if not self.save_traces or self.traces_path is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_file = self.traces_path / f"entailment_scoring_{timestamp}.json"
        
        trace_data = {
            "timestamp": timestamp,
            "method": "entailment",
            "model": self.client.model_name,
            "hypothesis_template": self.hypothesis_template,
            "predicate": predicate,
            "hypothesis": self.hypothesis_template.format(predicate=predicate),
            "nodes": [
                {
                    "id": node.get("id", ""),
                    "type": node.get("type", ""),
                    "name": node.get("name", ""),
                    "description": node.get("description", "")[:200],
                    "premise": node_infos[i] if i < len(node_infos) else "",
                    "score": scores[i] if i < len(scores) else 0.0
                }
                for i, node in enumerate(nodes)
            ],
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
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved entailment scoring trace to {trace_file}")





