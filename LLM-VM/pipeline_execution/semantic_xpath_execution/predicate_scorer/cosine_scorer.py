"""
Cosine similarity-based predicate scorer.

Uses TAS-B embeddings to score how well node descriptions match semantic predicates.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from client import get_tas_b_client, TASBClient, OpenAIEmbeddingClient
from client.embedding_cache import CachingEmbeddingClient, DEFAULT_CACHE_DIR
from .base import PredicateScorer, ScoringResult, BatchScoringResult


logger = logging.getLogger(__name__)


class CosinePredicateScorer(PredicateScorer):
    """
    Cosine similarity-based implementation of predicate scoring.
    
    Uses TAS-B model to generate embeddings and compute cosine similarity
    between node descriptions and semantic predicates.
    """
    
    DEFAULT_TRACES_PATH = Path(__file__).parent.parent / "traces" / "reasoning_traces"
    
    def __init__(
        self, 
        client: Optional[TASBClient] = None, 
        predicate_template: str = "{predicate}",
        save_traces: bool = True,
        traces_path: Path = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the cosine scorer.
        
        Args:
            client: Optional embedding client. If not provided, one will be created
                    based on the experiment config (cosine.model).
            predicate_template: Template for expanding predicate into a query.
                               Use {predicate} as placeholder.
            save_traces: Whether to save reasoning traces to disk.
            traces_path: Optional custom path for trace files. If None, traces are not saved.
            config: Full experiment config dict (from YAML). Used to read cosine.model
                    and route to either a local TAS-B model or an OpenAI/OpenRouter
                    embedding model (e.g., Qwen3-8B embed).
        """
        self._client = client
        self.predicate_template = predicate_template
        self.save_traces = save_traces
        self.traces_path = traces_path
        self._config = config
        
        # Only create traces directory if explicitly provided
        if self.traces_path:
            self.traces_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def client(self):
        """Lazy load the embedding client (with disk cache)."""
        if self._client is None:
            # If we have a full experiment config, prefer that path
            if self._config is not None:
                cosine_cfg = self._config.get("cosine", {})
                model_name = cosine_cfg.get(
                    "model",
                    "sentence-transformers/msmarco-distilbert-base-tas-b",
                )
                if model_name.startswith("sentence-transformers/"):
                    # Local HF model (TAS-B or compatible)
                    delegate = TASBClient(model_name=model_name)
                    self._client = CachingEmbeddingClient(
                        delegate,
                        cache_dir=DEFAULT_CACHE_DIR,
                        openai_config=None,
                    )
                else:
                    # Remote embedding model via OpenAI/OpenRouter.
                    # Prefer a cosine-specific openai block if provided; otherwise
                    # fall back to the top-level openai config.
                    openai_cfg = cosine_cfg.get("openai") or self._config.get("openai", {})
                    delegate = OpenAIEmbeddingClient(
                        model_name=model_name,
                        openai_config=openai_cfg,
                    )
                    self._client = CachingEmbeddingClient(
                        delegate,
                        cache_dir=DEFAULT_CACHE_DIR,
                        openai_config=openai_cfg,
                    )
            else:
                # Fallback to legacy TAS-B singleton
                delegate = get_tas_b_client()
                self._client = CachingEmbeddingClient(
                    delegate,
                    cache_dir=DEFAULT_CACHE_DIR,
                    openai_config=None,
                )
        return self._client
    
    def score_batch(
        self, 
        nodes: List[Dict[str, Any]], 
        predicate: str
    ) -> BatchScoringResult:
        """
        Score multiple nodes against a predicate using cosine similarity.
        
        Args:
            nodes: List of node dicts with 'id', 'type', 'description' keys
            predicate: The semantic predicate to match
            
        Returns:
            BatchScoringResult with scores for each node
        """
        if not nodes:
            return BatchScoringResult(predicate=predicate, results=[])
        
        # Build node description strings
        node_texts = []
        for node in nodes:
            node_type = node.get("type", "")
            name = node.get("name", "")
            description = node.get("description", "")
            
            # Construct informative text for embedding
            if name and description:
                text = f"{node_type}: {name} - {description}"
            elif description:
                text = f"{node_type}: {description}"
            elif name:
                text = f"{node_type}: {name}"
            else:
                text = f"{node_type}"
            
            node_texts.append(text)
        
        try:
            # Expand predicate with template
            query_text = self.predicate_template.format(predicate=predicate)
            
            # Get embeddings for all node texts
            node_embeddings = self.client.get_embeddings(node_texts, normalize=True)
            
            # Get embedding for predicate query
            predicate_embedding = self.client.get_embedding(query_text, normalize=True)
            
            # Compute cosine similarities (embeddings are already normalized)
            # similarity = dot product for normalized vectors
            similarities = np.dot(node_embeddings, predicate_embedding)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1, but typically 0 to 1 for semantic similarity)
            # We'll use (similarity + 1) / 2 to map [-1, 1] to [0, 1], but since these are
            # semantic embeddings, they're usually already in [0, 1] range
            scores = np.clip(similarities, 0, 1).tolist()
            
            # Build results
            results = []
            for i, node in enumerate(nodes):
                score = scores[i]
                results.append(ScoringResult(
                    node_id=node.get("id", f"node_{i}"),
                    node_type=node.get("type", ""),
                    node_description=node.get("description", ""),
                    predicate=predicate,
                    score=score,
                    reasoning=f"Cosine similarity: {score:.4f}"
                ))
            
            # Save trace if enabled
            if self.save_traces:
                self._save_trace(predicate, query_text, nodes, node_texts, scores, results)
            
            return BatchScoringResult(
                predicate=predicate,
                results=results,
                metadata={
                    "method": "cosine",
                    "predicate_template": self.predicate_template,
                    "model": self.client.model_name,
                    "embedding_dim": self.client.embedding_dim
                }
            )
            
        except Exception as e:
            logger.error(f"Error scoring batch with cosine similarity: {e}")
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
        query_text: str,
        nodes: List[Dict[str, Any]], 
        node_texts: List[str],
        scores: List[float],
        results: List[ScoringResult]
    ):
        """Save reasoning trace to disk."""
        if not self.save_traces or self.traces_path is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_file = self.traces_path / f"cosine_scoring_{timestamp}.json"
        
        trace_data = {
            "timestamp": timestamp,
            "method": "cosine",
            "model": self.client.model_name,
            "embedding_dim": self.client.embedding_dim,
            "predicate_template": self.predicate_template,
            "predicate": predicate,
            "query_text": query_text,
            "nodes": [
                {
                    "id": node.get("id", ""),
                    "type": node.get("type", ""),
                    "name": node.get("name", ""),
                    "description": node.get("description", "")[:200],
                    "text_for_embedding": node_texts[i] if i < len(node_texts) else "",
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
        
        logger.debug(f"Saved cosine scoring trace to {trace_file}")





