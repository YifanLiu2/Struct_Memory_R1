"""
Demo Logger - Tracks scoring details for visualization.

Provides detailed logging of:
- Child-to-parent score contributions in aggregation predicates
- Accumulated scores across query steps
- Score propagation through the tree hierarchy
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class ChildScoreContribution:
    """Represents a child node's score contribution to its parent."""
    child_name: str
    child_path: str
    child_type: str
    raw_score: float
    weight: float  # For agg_prev: subtree_size; for agg_exists: 1.0
    weighted_contribution: float  # raw_score * weight (or just raw_score for exists)
    is_best_match: bool = False


@dataclass 
class ParentScoreTrace:
    """Trace of how children contribute to a parent's aggregation score."""
    parent_name: str
    parent_path: str
    parent_type: str
    predicate_type: str  # 'agg_exists', 'agg_prev', 'and', 'or'
    formula: str
    child_contributions: List[ChildScoreContribution] = field(default_factory=list)
    aggregated_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parentName": self.parent_name,
            "parentPath": self.parent_path,
            "parentType": self.parent_type,
            "predicateType": self.predicate_type,
            "formula": self.formula,
            "childContributions": [
                {
                    "childName": c.child_name,
                    "childPath": c.child_path,
                    "childType": c.child_type,
                    "rawScore": round(c.raw_score, 4),
                    "weight": round(c.weight, 4),
                    "weightedContribution": round(c.weighted_contribution, 4),
                    "isBestMatch": c.is_best_match,
                }
                for c in self.child_contributions
            ],
            "aggregatedScore": round(self.aggregated_score, 4),
        }


@dataclass
class StepScoreEntry:
    """Score entry for a node at a specific step."""
    node_name: str
    node_path: str
    step_score: float  # Score from this step's predicate
    accumulated_score: float  # Product of all scores up to this step
    previous_accumulated: float  # Accumulated score before this step


@dataclass
class AccumulatedScoreTrace:
    """Trace of accumulated scores across steps for all nodes."""
    step_index: int
    step_predicate: str
    node_scores: List[StepScoreEntry] = field(default_factory=list)
    parent_contributions: List[ParentScoreTrace] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stepIndex": self.step_index,
            "stepPredicate": self.step_predicate,
            "nodeScores": [
                {
                    "nodeName": ns.node_name,
                    "nodePath": ns.node_path,
                    "stepScore": round(ns.step_score, 4),
                    "accumulatedScore": round(ns.accumulated_score, 4),
                    "previousAccumulated": round(ns.previous_accumulated, 4),
                }
                for ns in self.node_scores
            ],
            "parentContributions": [pc.to_dict() for pc in self.parent_contributions],
        }


class DemoLogger:
    """
    Logger for demo/visualization purposes.
    
    Tracks detailed score information across query execution:
    - How child scores contribute to parent scores (for agg predicates)
    - Accumulated scores at each step (previous_score * current_score)
    """
    
    def __init__(self, enabled: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self._logger = logging.getLogger("demo_logger")
        self._step_traces: List[AccumulatedScoreTrace] = []
        self._current_step: Optional[AccumulatedScoreTrace] = None
        self._accumulated_scores: Dict[str, float] = {}  # path -> accumulated score
        self._parent_contributions: List[ParentScoreTrace] = []
        
    def reset(self):
        """Reset all traces for a new query."""
        self._step_traces = []
        self._current_step = None
        self._accumulated_scores = {}
        self._parent_contributions = []
        
    def start_step(self, step_index: int, predicate_str: str):
        """Start tracking a new step."""
        if not self.enabled:
            return
        self._current_step = AccumulatedScoreTrace(
            step_index=step_index,
            step_predicate=predicate_str,
        )
        self._parent_contributions = []
        
    def end_step(self):
        """End the current step and save the trace."""
        if not self.enabled or not self._current_step:
            return
        self._current_step.parent_contributions = self._parent_contributions
        self._step_traces.append(self._current_step)
        self._current_step = None
        self._parent_contributions = []
        
    def log_node_score(
        self,
        node_name: str,
        node_path: str,
        step_score: float,
        previous_accumulated: Optional[float] = None,
    ):
        """Log a node's score for the current step and update accumulated score."""
        if not self.enabled or not self._current_step:
            return
            
        if previous_accumulated is None:
            previous_accumulated = self._accumulated_scores.get(node_path, 1.0)
        accumulated = previous_accumulated * step_score
        self._accumulated_scores[node_path] = accumulated
        
        self._current_step.node_scores.append(StepScoreEntry(
            node_name=node_name,
            node_path=node_path,
            step_score=step_score,
            accumulated_score=accumulated,
            previous_accumulated=previous_accumulated,
        ))
        
        if self.verbose:
            self._logger.info(
                f"[Step {self._current_step.step_index}] {node_name}: "
                f"step={step_score:.4f}, acc={accumulated:.4f} "
                f"(prev={previous_accumulated:.4f})"
            )
            
    def log_parent_contribution(
        self,
        parent_name: str,
        parent_path: str,
        parent_type: str,
        predicate_type: str,
        formula: str,
        child_contributions: List[Dict[str, Any]],
        aggregated_score: float,
    ):
        """Log how children contribute to a parent's score."""
        if not self.enabled:
            return
            
        contributions = []
        for c in child_contributions:
            # Use isBestMatch from input if provided, otherwise calculate
            is_best = c.get('isBestMatch', False)
            
            contributions.append(ChildScoreContribution(
                child_name=c.get('childName', ''),
                child_path=c.get('childPath', ''),
                child_type=c.get('childType', ''),
                raw_score=c.get('rawScore', 0),
                weight=c.get('weight', 1.0),
                weighted_contribution=c.get('weightedContribution', c.get('rawScore', 0)),
                is_best_match=is_best,
            ))
            
        trace = ParentScoreTrace(
            parent_name=parent_name,
            parent_path=parent_path,
            parent_type=parent_type,
            predicate_type=predicate_type,
            formula=formula,
            child_contributions=contributions,
            aggregated_score=aggregated_score,
        )
        
        self._parent_contributions.append(trace)
        
        if self.verbose:
            self._logger.info(
                f"[Parent] {parent_name} ({predicate_type}): {aggregated_score:.4f}"
            )
            for c in contributions:
                marker = " [BEST]" if c.is_best_match else ""
                self._logger.info(
                    f"  └── {c.child_name}: {c.raw_score:.4f} × {c.weight:.4f} = "
                    f"{c.weighted_contribution:.4f}{marker}"
                )
                
    def get_step_traces(self) -> List[Dict[str, Any]]:
        """Get all step traces as dictionaries."""
        return [t.to_dict() for t in self._step_traces]
    
    def get_accumulated_scores(self) -> Dict[str, float]:
        """Get current accumulated scores by path."""
        return dict(self._accumulated_scores)
    
    def to_json(self) -> str:
        """Export all traces as JSON."""
        return json.dumps({
            "timestamp": datetime.now().isoformat(),
            "stepTraces": self.get_step_traces(),
            "finalAccumulatedScores": {
                k: round(v, 4) for k, v in self._accumulated_scores.items()
            },
        }, indent=2)


# Global instance for easy access
_demo_logger: Optional[DemoLogger] = None


def get_demo_logger() -> DemoLogger:
    """Get the global demo logger instance."""
    global _demo_logger
    if _demo_logger is None:
        _demo_logger = DemoLogger(enabled=True, verbose=False)
    return _demo_logger


def demo_log(message: str, level: str = "info"):
    """Quick logging helper."""
    logger = get_demo_logger()
    if logger.enabled:
        log_func = getattr(logger._logger, level, logger._logger.info)
        log_func(f"[DEMO] {message}")
