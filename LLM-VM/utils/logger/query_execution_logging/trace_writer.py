"""
Trace Writer - Handles logging and trace file writing.

Writes comprehensive execution traces including:
- Stepwise traversal details
- Atomic scoring results
- Compound predicate composition (AND/OR)
- Hierarchical quantifier evaluation (exist/mass)
- Score fusion across steps (product)
- Final filtering decisions
- CRUD operation traces
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

from pipeline_execution.semantic_xpath_execution.execution_models import ExecutionResult
from pipeline_execution.semantic_xpath_execution.query_display import (
    canonicalize_query,
    canonicalize_parsed_ast_tree,
)

logger = logging.getLogger(__name__)


class TraceWriter:
    """
    Writes detailed reasoning traces to files.
    
    Produces JSON trace files for analysis.
    """
    
    def __init__(
        self,
        traces_path: Path = None
    ):
        """
        Initialize the trace writer.
        
        Args:
            traces_path: Directory for JSON trace files. If None, traces are not saved.
        """
        self.traces_path = traces_path
        
        # Only create directory if traces_path is explicitly provided
        if self.traces_path:
            self.traces_path.mkdir(parents=True, exist_ok=True)
    
    def save_traces(self, timestamp: str, result: ExecutionResult):
        """
        Save reasoning traces to file.
        
        Args:
            timestamp: Timestamp string for file naming
            result: ExecutionResult containing all trace data
        """
        if self.traces_path is None:
            return
        self._save_json_trace(timestamp, result)
    
    def _save_json_trace(self, timestamp: str, result: ExecutionResult):
        """
        Save detailed JSON trace for analysis.
        
        Structure:
        {
            "timestamp": "...",
            "query": "...",
            "data_file": "...",
            "execution_time_ms": ...,
            "traversal_steps": [...],
            "scoring_traces": [...],
            "scoring_details": {
                "step_scores": [...],
                "score_fusion": {...}
            },
            "final_filtering": {...},
            "matched_nodes": [...],
            "summary": {...}
        }
        """
        trace_file = self.traces_path / f"execution_{timestamp}.json"
        
        # Build scoring details from scoring traces
        step_scores = []
        for i, trace in enumerate(result.scoring_traces):
            step_detail = {
                "step_index": i,
                "predicate": trace.get("predicate", ""),
                "predicate_ast": trace.get("predicate_ast", {}),
                "node_scores": trace.get("node_scores", []),
                "ranking": trace.get("ranking", [])
            }
            step_scores.append(step_detail)
        
        # Build score fusion details
        score_fusion = None
        if result.score_fusion_trace:
            score_fusion = result.score_fusion_trace.to_dict()
        
        # Build final filtering details
        final_filtering = None
        if result.final_filtering_trace:
            final_filtering = result.final_filtering_trace.to_dict()
        
        # Build parsed AST if available
        parsed_ast = None
        parsed_ast_tree = None
        canonical_ast_tree = None
        if result.parsed_ast:
            parsed_ast = result.parsed_ast.to_dict()
            parsed_ast_tree = result.parsed_ast.to_tree_string()
            canonical_ast_tree = canonicalize_parsed_ast_tree(result.parsed_ast)

        canonical_query = canonicalize_query(result.query)
        
        trace_data = {
            "timestamp": timestamp,
            "query": result.query,
            "canonical_query": canonical_query,
            "data_file": result.data_file,
            "execution_time_ms": result.execution_time_ms,
            
            # Parsed AST (NEW: for debugging and visualization)
            "parsed_ast": parsed_ast,
            "parsed_ast_tree": parsed_ast_tree,
            "canonical_parsed_ast_tree": canonical_ast_tree,
            
            # Execution log (human-readable steps)
            "execution_log": result.execution_log,
            
            # Stepwise traversal
            "traversal_steps": [step.to_dict() for step in result.traversal_steps],
            
            # Raw scoring traces (detailed per-step)
            "scoring_traces": result.scoring_traces,
            
            # Structured scoring details
            "scoring_details": {
                "step_scores": step_scores,
                "score_fusion": score_fusion
            },
            
            # Final filtering
            "final_filtering": final_filtering,
            
            # Results
            "matched_nodes": [m.to_dict() for m in result.matched_nodes],
            
            # Summary statistics
            "summary": {
                "total_steps": len(result.traversal_steps),
                "total_scoring_calls": len(result.scoring_traces),
                "matched_count": len(result.matched_nodes),
                "execution_time_ms": result.execution_time_ms,
                "has_score_fusion": score_fusion is not None,
                "num_nodes_before_filter": (
                    result.final_filtering_trace.before_filter_count 
                    if result.final_filtering_trace else 0
                ),
                "num_nodes_after_filter": (
                    result.final_filtering_trace.after_filter_count 
                    if result.final_filtering_trace else len(result.matched_nodes)
                )
            }
        }
        
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved reasoning trace to {trace_file}")
        
        # Also print path to console for easy access
        print(f"[Trace saved] {trace_file}")
    
    # =========================================================================
    # CRUD Operation Trace Methods
    # =========================================================================
    
    def save_crud_traces(self, timestamp: str, result: Dict[str, Any]):
        """
        Save CRUD operation traces to file.
        
        Args:
            timestamp: Timestamp string for file naming
            result: CRUD operation result dictionary
        """
        if self.traces_path is None:
            return
        self._save_crud_json_trace(timestamp, result)
    
    def _save_crud_json_trace(self, timestamp: str, result: Dict[str, Any]):
        """
        Save detailed CRUD operation JSON trace.
        
        Structure:
        {
            "timestamp": "...",
            "operation": "...",
            "user_query": "...",
            "full_query": "...",
            "success": ...,
            "intent": {...},
            "xpath_query": "...",
            "reasoning_trace": {...},
            "operation_specific": {...},
            "tree_version": {...}
        }
        """
        operation = result.get("operation", "unknown").lower()
        trace_file = self.traces_path / f"crud_{operation}_{timestamp}.json"
        
        xpath_query = result.get("xpath_query")
        canonical_xpath_query = canonicalize_query(xpath_query) if xpath_query else None
        operation = result.get("operation")
        canonical_full_query = None
        if canonical_xpath_query and operation:
            canonical_full_query = f"{operation}({canonical_xpath_query})"

        trace_data = {
            "timestamp": result.get("timestamp", timestamp),
            "operation": result.get("operation"),
            "user_query": result.get("user_query"),
            "full_query": result.get("full_query"),
            "xpath_query": result.get("xpath_query"),
            "canonical_full_query": canonical_full_query,
            "canonical_xpath_query": canonical_xpath_query,
            "success": result.get("success"),
            
            "intent": result.get("intent"),
            "reasoning_trace": result.get("reasoning_trace"),
            
            # Operation-specific data
            "operation_data": self._extract_operation_data(result),
            
            # Tree version
            "tree_version": result.get("tree_version"),
            
            # Summary
            "summary": {
                "operation": result.get("operation"),
                "success": result.get("success"),
                "affected_count": len(result.get("affected_nodes", result.get("deleted_paths", result.get("updated_paths", []))))
            }
        }
        
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved CRUD trace to {trace_file}")
        print(f"[CRUD trace saved] {trace_file}")
    
    def _extract_operation_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract operation-specific data for the trace."""
        operation = result.get("operation", "")
        
        if operation == "READ":
            return {
                "candidates_count": result.get("candidates_count"),
                "selected_count": result.get("selected_count"),
                "selected_nodes": result.get("selected_nodes")
            }
        elif operation == "DELETE":
            return {
                "deleted_count": result.get("deleted_count"),
                "deleted_paths": result.get("deleted_paths"),
                "deletion_results": result.get("deletion_results")
            }
        elif operation == "UPDATE":
            return {
                "updated_count": result.get("updated_count"),
                "updated_paths": result.get("updated_paths"),
                "update_results": result.get("update_results")
            }
        elif operation == "CREATE":
            return {
                "created_path": result.get("created_path"),
                "insert_result": result.get("insert_result"),
                "content_result": result.get("content_result"),
                "insertion_point": result.get("insertion_point")
            }
        
        return {}
