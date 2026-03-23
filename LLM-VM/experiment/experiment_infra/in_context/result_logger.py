"""
Result Logger - Formats and saves in-context evaluation results.

Provides utilities for:
- Per-query result files (query_result.json + reasoning_traces/)
- Session result files (one per session)
- Experiment-level logs (query_id -> related node paths)
"""

import json
import difflib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """Result for a single query within a session."""
    query_id: str
    user_query: str
    selected_nodes: List[str]
    related_nodes: List[str]
    downstream_task_result: Dict[str, Any]
    execution_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "user_query": self.user_query,
            "selected_nodes": self.selected_nodes,
            "related_nodes": self.related_nodes,
            "downstream_task_result": self.downstream_task_result,
            "execution_time_ms": round(self.execution_time_ms, 2)
        }


@dataclass
class SessionResult:
    """Result for a complete session."""
    session_id: str
    queries: List[QueryResult] = field(default_factory=list)
    final_tree_version: int = 1
    total_execution_time_ms: float = 0.0
    
    def add_query_result(self, result: QueryResult):
        """Add a query result to the session."""
        self.queries.append(result)
        self.total_execution_time_ms += result.execution_time_ms
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "queries": [q.to_dict() for q in self.queries],
            "final_tree_version": self.final_tree_version,
            "total_execution_time_ms": round(self.total_execution_time_ms, 2)
        }


class InContextResultLogger:
    """
    Logger for in-context evaluation results.
    
    Handles:
    - Per-query result files (query_result.json in each query directory)
    - Per-query reasoning traces (raw LLM response in reasoning_traces/)
    - Session result files (one JSON file per session)
    - Experiment-level summary log (all query results)
    """
    
    def __init__(self, experiment_dir: Path):
        """
        Initialize the result logger.
        
        Args:
            experiment_dir: Base directory for experiment results
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Track all results for experiment log
        self._all_results: Dict[str, List[str]] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}
    
    def create_session_dir(self, session_id: str) -> Path:
        """
        Create directory for a session.
        
        Args:
            session_id: Session identifier (e.g., "Session_1")
            
        Returns:
            Path to session directory
        """
        session_dir = self.experiment_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    @staticmethod
    def _compute_xml_diff(input_xml: str, result_xml: str) -> str:
        """
        Compute a unified diff between input XML and the LLM's returned XML.
        
        Args:
            input_xml: The XML tree before the query
            result_xml: The XML returned by the LLM
            
        Returns:
            Unified diff string showing additions (+), removals (-), and context
        """
        input_lines = input_xml.splitlines(keepends=True)
        result_lines = result_xml.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            input_lines,
            result_lines,
            fromfile="input_tree.xml",
            tofile="result_tree.xml",
            lineterm=""
        )
        return "\n".join(diff)
    
    def save_query_result(
        self,
        query_dir: Path,
        user_query: str,
        pipeline_result,  # InContextResult
        input_xml: str = "",
        cost_usd: float = 0.0
    ):
        """
        Save per-query result files: query_result.json and reasoning_traces/.
        
        Creates:
            query_dir/
            ├── query_result.json       # Main result with all fields + xml_diff
            └── reasoning_traces/
                └── llm_response_{timestamp}.json  # Raw LLM response
        
        Args:
            query_dir: Directory for this query
            user_query: Original user query
            pipeline_result: InContextResult from pipeline
            input_xml: The XML tree state before this query (for diff computation)
            cost_usd: Calculated cost for this query
        """
        query_dir.mkdir(parents=True, exist_ok=True)
        
        # Build query result
        token_usage = pipeline_result.token_usage if pipeline_result.token_usage else {}
        result_xml = pipeline_result.result_xml or ""
        
        # Compute diff between input XML and the LLM's returned XML
        xml_diff = ""
        if input_xml and result_xml:
            xml_diff = self._compute_xml_diff(input_xml, result_xml)
        
        query_result = {
            "user_query": user_query,
            "operation": pipeline_result.operation,
            "success": pipeline_result.success,
            "version_used": pipeline_result.version_used,
            "reasoning": pipeline_result.reasoning,
            "selected_nodes": pipeline_result.selected_nodes,
            "related_nodes": pipeline_result.related_nodes,
            "result_xml": result_xml,
            "xml_diff": xml_diff,
            "execution_time_ms": round(pipeline_result.execution_time_ms, 2),
            "token_usage": token_usage,
            "cost_usd": round(cost_usd, 5),
            "error": pipeline_result.error
        }
        
        # Save query_result.json
        with open(query_dir / "query_result.json", "w", encoding="utf-8") as f:
            json.dump(query_result, f, indent=2, ensure_ascii=False)
        
        # Save reasoning traces (raw LLM response)
        traces_dir = query_dir / "reasoning_traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_data = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "operation": pipeline_result.operation,
            "raw_response": pipeline_result.raw_response,
            "token_usage": token_usage
        }
        
        trace_path = traces_dir / f"llm_response_{timestamp}.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
    
    def save_session_result(self, session_result: SessionResult, session_dir: Path):
        """
        Save session result to file.
        
        Args:
            session_result: SessionResult object
            session_dir: Directory to save to
        """
        output_path = session_dir / "session_result.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(session_result.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Track for experiment log
        for query in session_result.queries:
            query_key = f"{session_result.session_id}_{query.query_id}"
            self._all_results[query_key] = query.selected_nodes
            
            # Extract token usage from downstream result
            token_usage = query.downstream_task_result.get("token_usage", {})
            
            self._session_data[query_key] = {
                "query": query.user_query,
                "operation": query.downstream_task_result.get("operation", "UNKNOWN"),
                "success": query.downstream_task_result.get("success", False),
                "selected_nodes": query.selected_nodes,
                "top_k_nodes": query.related_nodes,
                "execution_time_ms": query.execution_time_ms,
                "token_usage": token_usage
            }
    
    def save_experiment_log(
        self,
        experiment_name: str,
        total_sessions: int,
        total_execution_time_ms: float
    ):
        """
        Save experiment-level summary log.
        
        Args:
            experiment_name: Name of the experiment
            total_sessions: Total number of sessions
            total_execution_time_ms: Total execution time
        """
        experiment_log = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_sessions": total_sessions,
            "total_queries": len(self._all_results),
            "results": self._session_data,
            "node_paths_summary": self._all_results,
            "total_execution_time_ms": round(total_execution_time_ms, 2)
        }
        
        output_path = self.experiment_dir / "Experiment_Log.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(experiment_log, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def format_query_result(
        query_index: int,
        user_query: str,
        pipeline_result,  # InContextResult
    ) -> QueryResult:
        """
        Format a pipeline result into a QueryResult.
        
        Args:
            query_index: 1-based query index
            user_query: Original user query
            pipeline_result: InContextResult from pipeline
            
        Returns:
            QueryResult object
        """
        return QueryResult(
            query_id=f"Query_{query_index:03d}",
            user_query=user_query,
            selected_nodes=pipeline_result.selected_nodes,
            related_nodes=pipeline_result.related_nodes,
            downstream_task_result={
                "operation": pipeline_result.operation,
                "success": pipeline_result.success,
                "version_used": pipeline_result.version_used,
                "reasoning": pipeline_result.reasoning,
                "token_usage": pipeline_result.token_usage,
                "error": pipeline_result.error
            },
            execution_time_ms=pipeline_result.execution_time_ms
        )
    
    def save_multi_turn_summary(
        self,
        multi_turn_dir: Path,
        multi_turn_id: str,
        turn_results: List[Dict[str, Any]],
        final_tree_version: int = 1,
    ):
        """
        Save a per-multi-turn-conversation summary with information from each turn.
        
        Creates a session_result.json inside the multi-turn folder that contains
        an overall summary (total time, tokens, cost) and detailed per-turn breakdown.
        
        Args:
            multi_turn_dir: Directory for this multi-turn conversation
            multi_turn_id: Identifier for this multi-turn conversation (e.g. "multi_turn_1")
            turn_results: List of per-turn result dicts, each containing:
                - turn_index, user_query, operation, success, version_used, reasoning,
                  selected_nodes, related_nodes, execution_time_ms, token_usage, cost_usd, error
            final_tree_version: Final version number of the tree after all turns
        """
        total_turns = len(turn_results)
        total_time_ms = sum(t.get("execution_time_ms", 0) for t in turn_results)
        total_prompt = sum(t.get("token_usage", {}).get("prompt_tokens", 0) for t in turn_results)
        total_completion = sum(t.get("token_usage", {}).get("completion_tokens", 0) for t in turn_results)
        total_tokens = sum(t.get("token_usage", {}).get("total_tokens", 0) for t in turn_results)
        total_cost = sum(t.get("cost_usd", 0) for t in turn_results)
        
        successful_turns = sum(1 for t in turn_results if t.get("success", False))
        
        summary = {
            "multi_turn_id": multi_turn_id,
            "total_turns": total_turns,
            "successful_turns": successful_turns,
            "final_tree_version": final_tree_version,
            "summary": {
                "total_execution_time_ms": round(total_time_ms, 2),
                "total_execution_time_s": round(total_time_ms / 1000, 2),
                "total_tokens": {
                    "prompt": total_prompt,
                    "completion": total_completion,
                    "total": total_tokens
                },
                "total_cost_usd": round(total_cost, 5),
                "avg_execution_time_ms": round(total_time_ms / total_turns, 2) if total_turns > 0 else 0,
                "avg_tokens_per_turn": round(total_tokens / total_turns, 2) if total_turns > 0 else 0,
                "avg_cost_per_turn_usd": round(total_cost / total_turns, 5) if total_turns > 0 else 0,
            },
            "turns": turn_results
        }
        
        output_path = multi_turn_dir / "session_result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def format_downstream_result(pipeline_result) -> Dict[str, Any]:
        """
        Format downstream task result.
        
        Args:
            pipeline_result: InContextResult from pipeline
            
        Returns:
            Dict with downstream task details
        """
        return {
            "operation": pipeline_result.operation,
            "success": pipeline_result.success,
            "version_used": pipeline_result.version_used,
            "reasoning": pipeline_result.reasoning,
            "result_preview": pipeline_result.result_xml[:500] if pipeline_result.result_xml else "",
            "token_usage": pipeline_result.token_usage,
            "error": pipeline_result.error
        }
