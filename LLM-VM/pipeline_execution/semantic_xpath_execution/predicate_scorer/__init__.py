import yaml
from pathlib import Path

from .base import PredicateScorer, ScoringResult, BatchScoringResult
from .llm_scorer import LLMPredicateScorer
from .entailment_scorer import EntailmentPredicateScorer
from .cosine_scorer import CosinePredicateScorer


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_scorer(method: str = None, config: dict = None, traces_path: Path = None) -> PredicateScorer:
    """
    Factory function to create the appropriate scorer based on config.
    
    Args:
        method: Scoring method ("llm", "entailment", or "cosine"). 
                If None, uses value from config.yaml.
        config: Optional config dict. If not provided, loads from config.yaml.
        traces_path: Optional custom path for trace files.
    
    Returns:
        PredicateScorer instance
    """
    if config is None:
        config = load_config()
    
    executor_config = config.get("xpath_executor", {})
    
    if method is None:
        method = executor_config.get("scoring_method", "llm")
    
    if method == "entailment":
        entailment_config = config.get("entailment", {})
        hypothesis_template = entailment_config.get(
            "hypothesis_template", 
            "This is related to {predicate}."
        )
        model_name = entailment_config.get("model", "facebook/bart-large-mnli")
        
        return EntailmentPredicateScorer(
            hypothesis_template=hypothesis_template,
            model_name=model_name,
            traces_path=traces_path
        )
    elif method == "cosine":
        cosine_config = config.get("cosine", {})
        predicate_template = cosine_config.get(
            "predicate_template",
            "{predicate}"
        )
        return CosinePredicateScorer(
            predicate_template=predicate_template,
            traces_path=traces_path,
            config=config,
        )
    else:
        # Default to LLM
        return LLMPredicateScorer(traces_path=traces_path)


__all__ = [
    "PredicateScorer", 
    "ScoringResult",
    "BatchScoringResult",
    "LLMPredicateScorer", 
    "EntailmentPredicateScorer",
    "CosinePredicateScorer",
    "get_scorer"
]
