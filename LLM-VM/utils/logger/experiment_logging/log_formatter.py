"""
Log Formatter - Utilities for formatting query master logs and experiment logs.

Provides functions to extract and structure data from pipeline results
into the query_master_log.json format.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class LogFormatter:
    """Formats pipeline results into structured log files."""
    
    @staticmethod
    def format_query_master_log(
        user_query: str,
        pipeline_result: Dict[str, Any],
        traces_dir: Path
    ) -> Dict[str, Any]:
        """
        Format pipeline result into query master log structure.
        
        Args:
            user_query: The original user query
            pipeline_result: Result dict from SemanticXPathPipeline.process_request()
            traces_dir: Directory containing trace files
            
        Returns:
            Dict with query master log structure
        """
        # Extract version info
        version_xpath = LogFormatter._extract_version_xpath(pipeline_result)
        version_result = LogFormatter._extract_version_result(pipeline_result)
        
        # Extract operation type
        operation_type = pipeline_result.get("operation", "UNKNOWN")
        
        # Extract semantic xpath query from parsed_query (prefer canonical if available)
        semantic_xpath_query = pipeline_result.get("canonical_xpath_query")
        if not semantic_xpath_query:
            parsed_query = pipeline_result.get("parsed_query", {})
            semantic_xpath_query = parsed_query.get("xpath", "")
        
        # Find trace files
        trace_files = LogFormatter.get_trace_file_paths(traces_dir)
        
        # Extract semantic xpath results
        semantic_xpath_result = LogFormatter._extract_xpath_results(pipeline_result)
        
        # Extract downstream task result
        downstream_result = LogFormatter._extract_downstream_result(pipeline_result)
        
        return {
            "user_query": user_query,
            "version_xpath": version_xpath,
            "version_result": version_result,
            "operation_type": operation_type,
            "semantic_xpath_query": semantic_xpath_query,
            "semantic_xpath_execution_log": {
                "aggregation_execution_log": trace_files.get("execution"),
                "predicate_batch_scoring_log": trace_files.get("scoring")
            },
            "semantic_xpath_result": semantic_xpath_result,
            "downstream_task_result": downstream_result
        }
    
    @staticmethod
    def _extract_version_xpath(result: Dict[str, Any]) -> str:
        """Extract version xpath selector from result."""
        version_info = result.get("version_resolution", {})
        selector_type = version_info.get("selector_type", "AT")
        index = version_info.get("index", -1)
        semantic = version_info.get("semantic_query")
        
        if selector_type == "BEFORE":
            if semantic:
                return f'before(sem(content ~= "{semantic}"))'
            return f"before([{index}])"
        else:  # AT
            if semantic:
                return f'at(sem(content ~= "{semantic}"))'
            if index is not None:
                return f"at([{index}])"
            return "at([-1])"
    
    @staticmethod
    def _extract_version_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract version resolution details."""
        version_info = result.get("version_resolution", {})
        
        # Try to get resolved version from different fields
        resolved_version = result.get("version_used")
        if resolved_version is None:
            resolved_version = 1
        # Handle string version numbers
        if isinstance(resolved_version, str):
            try:
                resolved_version = int(resolved_version)
            except ValueError:
                resolved_version = 1
        
        return {
            "selector_type": version_info.get("selector_type", "AT"),
            "index": version_info.get("index", -1),
            "semantic_query": version_info.get("semantic_query"),
            "task_query": version_info.get("task_query"),
            "resolved_version": resolved_version
        }
    
    @staticmethod
    def _extract_xpath_results(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract semantic xpath execution results."""
        xpath_results = []
        
        # For READ operations, get selected nodes
        selected_nodes = result.get("selected_nodes", [])
        if selected_nodes:
            for node in selected_nodes:
                xpath_results.append({
                    "node_relative_path": node.get("tree_path", node.get("path", "")),
                    "node_info": {
                        "name": node.get("name", ""),
                        "type": node.get("type", "")
                    },
                    "children_names": LogFormatter._get_children_names(node),
                    "score": node.get("score", node.get("relevance_score", 0.0))
                })
        
        # For CUD operations, get candidate nodes from xpath execution
        candidates = result.get("candidates", [])
        if candidates and not xpath_results:
            for candidate in candidates:
                xpath_results.append({
                    "node_relative_path": candidate.get("tree_path", candidate.get("path", "")),
                    "node_info": {
                        "name": candidate.get("name", ""),
                        "type": candidate.get("type", "")
                    },
                    "children_names": LogFormatter._get_children_names(candidate),
                    "score": candidate.get("score", 0.0)
                })
        
        return xpath_results
    
    @staticmethod
    def _get_children_names(node: Dict[str, Any]) -> List[str]:
        """Extract children names from a node."""
        children = node.get("children", [])
        if isinstance(children, list):
            return [
                child.get("name", child.get("tag", "")) 
                for child in children 
                if isinstance(child, dict)
            ]
        return []
    
    @staticmethod
    def _extract_downstream_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract downstream task (CRUD operation) result."""
        operation = result.get("operation", "UNKNOWN")
        downstream = {
            "operation": operation,
            "success": result.get("success", False)
        }
        
        if operation == "READ":
            downstream["selected_count"] = result.get("selected_count", 0)
            downstream["candidates_count"] = result.get("candidates_count", 0)
        
        elif operation == "DELETE":
            downstream["deleted_paths"] = result.get("deleted_paths", [])
            downstream["deleted_count"] = result.get("deleted_count", 0)
        
        elif operation == "UPDATE":
            downstream["updated_paths"] = result.get("updated_paths", [])
            downstream["updated_count"] = result.get("updated_count", 0)
            # Simplify update_results for logging
            update_results = result.get("update_results", [])
            downstream["update_summary"] = [
                {"path": u.get("path", ""), "success": u.get("success", False)}
                for u in update_results
            ]
        
        elif operation == "CREATE":
            downstream["created_path"] = result.get("created_path")
            insertion_point = result.get("insertion_point", {})
            downstream["insertion_point"] = {
                "parent_path": insertion_point.get("parent_path"),
                "position": insertion_point.get("position")
            }
            content_result = result.get("content_result", {})
            downstream["created_node_type"] = content_result.get("node_type")
        
        # Add timing info if available
        timing = result.get("timing", {})
        if timing:
            downstream["total_time_ms"] = timing.get("total_time_ms")
        
        return downstream
    
    @staticmethod
    def get_trace_file_paths(traces_dir: Path) -> Dict[str, Optional[str]]:
        """
        Find trace file paths in the traces directory.
        
        Args:
            traces_dir: Directory to search for trace files
            
        Returns:
            Dict with keys 'execution' and 'scoring' mapping to relative file paths
        """
        result = {
            "execution": None,
            "scoring": None
        }
        
        reasoning_traces = traces_dir / "reasoning_traces"
        if not reasoning_traces.exists():
            return result
        
        for file in reasoning_traces.iterdir():
            if file.is_file() and file.suffix == ".json":
                filename = file.name
                relative_path = f"reasoning_traces/{filename}"
                
                if filename.startswith("execution_"):
                    result["execution"] = relative_path
                elif filename.startswith("entailment_scoring_") or filename.startswith("scoring_"):
                    # May have multiple scoring files, get the latest
                    if result["scoring"] is None or filename > result["scoring"]:
                        result["scoring"] = relative_path
        
        return result
    
    @staticmethod
    def extract_top_k_paths(result: Dict[str, Any], k: int = 5) -> List[str]:
        """
        Extract top-k node relative paths from pipeline result.
        
        Args:
            result: Pipeline result dict
            k: Maximum number of paths to return
            
        Returns:
            List of node relative paths
        """
        paths = []
        
        # Try selected_nodes first (READ operations)
        selected_nodes = result.get("selected_nodes", [])
        for node in selected_nodes[:k]:
            path = node.get("tree_path", node.get("path", ""))
            if path:
                paths.append(path)
        
        # If no selected nodes, try candidates
        if not paths:
            candidates = result.get("candidates", [])
            for candidate in candidates[:k]:
                path = candidate.get("tree_path", candidate.get("path", ""))
                if path:
                    paths.append(path)
        
        # For CUD operations, include the affected paths
        if not paths:
            # DELETE
            deleted_paths = result.get("deleted_paths", [])
            paths.extend(deleted_paths[:k])
            
            # CREATE
            created_path = result.get("created_path")
            if created_path:
                paths.append(created_path)
            
            # UPDATE
            updated_paths = result.get("updated_paths", [])
            paths.extend(updated_paths[:k])
        
        return paths[:k]
    
    @staticmethod
    def format_experiment_log(
        experiment_name: str,
        sessions_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format experiment-level summary log.
        
        Args:
            experiment_name: Name of the experiment (e.g., "experiment_001")
            sessions_data: Dict mapping session/query IDs to their results
            
        Returns:
            Experiment log dict
        """
        total_sessions = len(set(
            key.split("_Query_")[0] for key in sessions_data.keys()
        ))
        total_queries = len(sessions_data)
        
        return {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_sessions": total_sessions,
            "total_queries": total_queries,
            "results": sessions_data
        }
    
    @staticmethod
    def save_query_master_log(log_data: Dict[str, Any], output_path: Path):
        """Save query master log to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def save_experiment_log(log_data: Dict[str, Any], output_path: Path):
        """Save experiment log to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def format_token_usage(token_data: Dict[str, Any]) -> str:
        """
        Format token usage data as a human-readable string.
        
        Args:
            token_data: Token usage dict with prompt_tokens, completion_tokens, etc.
            
        Returns:
            Formatted string representation
        """
        if not token_data:
            return "No token data available"
        
        lines = []
        lines.append(f"Total: {token_data.get('total_tokens', 0):,} tokens")
        lines.append(f"  Prompt: {token_data.get('prompt_tokens', 0):,}")
        lines.append(f"  Completion: {token_data.get('completion_tokens', 0):,}")
        
        if "cost_usd" in token_data:
            lines.append(f"  Cost: ${token_data['cost_usd']:.4f}")
        
        if "by_stage" in token_data and token_data["by_stage"]:
            lines.append("  By Stage:")
            for stage, usage in token_data["by_stage"].items():
                lines.append(f"    {stage}: {usage.get('total_tokens', 0):,} tokens")
        
        return "\n".join(lines)
