"""
Semantic XPath Experiment Runner - Execute queries across sessions with detailed tracking.

Usage:
    python -m experiment.experiment_infra.semantic_xpath.semantic_xpath_experiment_runner --config experiment.yaml

Output Structure:
    experiment/experiment_result/semantic_xpath/{experiment_name}/
    ├── Experiment_Log.json
    ├── Cost_Summary.json           # Token costs and latency breakdown
    ├── experiment_config.yaml
    ├── {session_name}/             # Named sessions (e.g., read_queries/)
    │   ├── tree.xml
    │   └── Query_001/
    │       ├── query_master_log.json
    │       └── reasoning_traces/
    │           ├── execution_*.json
    │           └── entailment_scoring_*.json
    └── {session_name}/
        └── ...
"""

import json
import yaml
import shutil
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pipeline.semantic_xpath_pipeline import SemanticXPathPipeline
from utils.logger.experiment_logging import LogFormatter


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Token pricing (GPT-4o as default, update as needed)
PRICING = {
    "gpt-4o": {"prompt": 0.0025 / 1000, "completion": 0.01 / 1000},  # per token
    "gpt-4o-mini": {"prompt": 0.00015 / 1000, "completion": 0.0006 / 1000},
    "gpt-5": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
    "o1": {"prompt": 0.015 / 1000, "completion": 0.06 / 1000},
    "o3": {"prompt": 0.015 / 1000, "completion": 0.06 / 1000},
}


class ExperimentRunner:
    """
    Runs experiments across multiple sessions with session-based query execution.
    
    Each session:
    - Contains a list of queries
    - Controlled by `use_versions` flag:
        - false (default): Each query runs independently on the original tree from storage.
          The tree is reset before each query so no CUD changes carry over.
        - true: CUD operations create new tree versions. Each consecutive query
          runs on the updated tree, so changes accumulate across queries.
    - Uses in-tree versioning when use_versions is true (changes patched to single tree.xml)
    - Produces query_master_log.json for each query
    
    Experiment-level output:
    - Experiment_Log.json summarizing all query results
    - Cost_Summary.json with token usage and cost breakdowns
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the experiment runner.
        
        Args:
            config_path: Path to experiment.yaml config file
        """
        # Resolve config path relative to project root if not absolute
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        self.config_path = config_path
        self.config = self._load_config()
        
        self.experiment_name = self.config.get("name", "experiment")
        self.sessions = self._parse_sessions(self.config.get("queries", []))
        
        # Load config from embedded 'config' section (required)
        embedded_config = self.config.get("config")
        if not embedded_config:
            raise ValueError(
                "Experiment config must have a 'config' section with all settings. "
                "See experiment.yaml for the expected format."
            )
        self.app_config = embedded_config
        self.model = embedded_config.get("openai", {}).get("model", "gpt-4o")
        
        # Extract scoring method and model from config
        executor_config = embedded_config.get("xpath_executor", {})
        self.scoring_method = executor_config.get("scoring_method", "llm")
        
        # Extract the model name used for scoring based on the method
        if self.scoring_method == "entailment":
            entailment_config = embedded_config.get("entailment", {})
            self.scoring_model = entailment_config.get("model", "facebook/bart-large-mnli")
        elif self.scoring_method == "cosine":
            cosine_config = embedded_config.get("cosine", {})
            self.scoring_model = cosine_config.get("model", "sentence-transformers/msmarco-distilbert-base-tas-b")
            # Also check if OpenRouter/OpenAI config is specified
            self.scoring_api = None
            if cosine_config.get("openai"):
                self.scoring_api = "OpenRouter" if "openrouter" in cosine_config.get("openai", {}).get("base_url", "").lower() else "OpenAI"
        else:  # llm
            self.scoring_model = embedded_config.get("openai", {}).get("model", "gpt-4o")
            self.scoring_api = "OpenAI"
        
        # Setup output directory
        self.base_output_dir = PROJECT_ROOT / "experiment" / "experiment_result" / "semantic_xpath"
        self.experiment_dir = None  # Set during run()
        
        # Track results for experiment log
        self._session_results: Dict[str, Dict[str, Any]] = {}
        
        # Track costs and tokens
        self._cost_tracker: List[Dict[str, Any]] = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate experiment config."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        if "queries" not in config or not config["queries"]:
            raise ValueError("Experiment config must have non-empty 'queries' list")
        
        return config
    
    def _parse_sessions(self, queries_config: list) -> List[Dict[str, Any]]:
        """
        Parse the queries config into a list of named sessions.
        
        Supports two formats:
          1. List format (unnamed):  ["query1", "query2"]  -> auto-named Session_N
          2. Dict format (named):    {name: "read_queries", queries: ["q1", "q2"], use_versions: "false"}
        
        The `use_versions` field controls tree state behavior:
          - 'false' (default): Each query runs independently on the original tree.
          - 'true': CUD operations create new versions; consecutive queries see updates.
        
        Returns:
            List of dicts with 'name', 'queries', 'use_versions', and optional 'tree_path' keys
        """
        sessions = []
        unnamed_counter = 1
        
        for i, entry in enumerate(queries_config):
            if isinstance(entry, dict):
                # Named session: {name: "...", queries: [...], use_versions: "false"}
                name = entry.get("name")
                query_list = entry.get("queries", [])
                # Parse use_versions: accepts string 'true'/'false' or bool, defaults to False
                use_versions_raw = entry.get("use_versions", "false")
                if isinstance(use_versions_raw, str):
                    use_versions = use_versions_raw.strip().lower() == "true"
                else:
                    use_versions = bool(use_versions_raw)
                
                if not name:
                    raise ValueError(f"Session dict at index {i} must have a 'name' key")
                if not isinstance(query_list, list) or not query_list:
                    print(f"  [SKIP] Session '{name}' has no queries, skipping.")
                    continue
                tree_path = entry.get("tree_path")
                sessions.append({"name": name, "queries": query_list, "use_versions": use_versions, "tree_path": tree_path})
            elif isinstance(entry, list):
                # Unnamed session: auto-name as Session_N, default use_versions=False
                if not entry:
                    raise ValueError(f"Query list at index {i} is empty")
                sessions.append({"name": f"Session_{unnamed_counter}", "queries": entry, "use_versions": False})
                unnamed_counter += 1
            else:
                raise ValueError(
                    f"Query entry at index {i} must be a list or dict. "
                    f"Use [\"query\"] for unnamed or {{name: \"...\", queries: [...]}} for named."
                )
        
        return sessions
    
    def _setup_experiment_dir(self) -> Path:
        """
        Create experiment directory using the experiment name.
        
        Uses the 'name' field from experiment.yaml as the folder name.
        If a folder with that name already exists, appends _2, _3, etc.
        """
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize experiment name for use as folder name
        safe_name = re.sub(r'[^\w\s-]', '', self.experiment_name)
        safe_name = re.sub(r'\s+', '_', safe_name).strip('_')
        if not safe_name:
            safe_name = "experiment"
        
        # Check for existing directory, add suffix if needed
        experiment_dir = self.base_output_dir / safe_name
        if experiment_dir.exists():
            counter = 2
            while (self.base_output_dir / f"{safe_name}_{counter}").exists():
                counter += 1
            experiment_dir = self.base_output_dir / f"{safe_name}_{counter}"
        
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy experiment config
        shutil.copy(self.config_path, experiment_dir / "experiment_config.yaml")
        
        return experiment_dir
    
    def _sanitize_query_name(self, query: str, index: int) -> str:
        """Create a sanitized folder name from query text, prefixed with index for ordering."""
        # Take first few words
        words = query.split()[:4]
        name = "_".join(words)
        
        # Remove special characters
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"\s+", "_", name)
        name = name[:30]  # Limit length
        
        if not name:
            name = f"{index:02d}_Query"
        else:
            name = f"{index:02d}_Query_{name}"
        
        return name
    
    def _calculate_cost(self, token_usage: Dict[str, int]) -> float:
        """
        Calculate cost in USD for the given token usage.
        
        Args:
            token_usage: Dict with prompt_tokens and completion_tokens
            
        Returns:
            Cost in USD
        """
        # Find pricing for current model
        pricing = None
        model_lower = self.model.lower()
        for model_key in PRICING:
            if model_lower.startswith(model_key):
                pricing = PRICING[model_key]
                break
        
        if not pricing:
            # Default to gpt-4o pricing
            pricing = PRICING["gpt-4o"]
        
        prompt_cost = token_usage.get("prompt_tokens", 0) * pricing["prompt"]
        completion_cost = token_usage.get("completion_tokens", 0) * pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def _extract_token_usage(self, result: Dict[str, Any], traces_dir: Path) -> Dict[str, Any]:
        """
        Extract token usage from pipeline result timing data and trace files.
        
        Args:
            result: Pipeline result dict
            traces_dir: Directory containing trace files
            
        Returns:
            Dict with token breakdown and cost
        """
        token_breakdown = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "by_stage": {}
        }
        
        # FIRST: Check if timing field has token data (from PipelineTimer)
        timing_data = result.get("timing", {})
        
        # Check total_tokens from timing (this aggregates all stages)
        total_tokens = timing_data.get("total_tokens")
        if total_tokens and total_tokens.get("total_tokens", 0) > 0:
            token_breakdown["prompt_tokens"] = total_tokens.get("prompt_tokens", 0)
            token_breakdown["completion_tokens"] = total_tokens.get("completion_tokens", 0)
            token_breakdown["total_tokens"] = total_tokens.get("total_tokens", 0)
        
        # Extract per-stage token usage from timing.stages
        stages = timing_data.get("stages", [])
        for stage in stages:
            stage_name = stage.get("name")
            stage_tokens = stage.get("token_usage")
            
            if stage_tokens and stage_name:
                # Map stage names to readable names
                if "generation" in stage_name.lower() or "query" in stage_name.lower():
                    display_name = "xpath_generation"
                elif "version" in stage_name.lower() and "resolution" in stage_name.lower():
                    display_name = "version_resolution"
                elif "xpath_execution" in stage_name.lower():
                    display_name = "xpath_execution"
                elif "downstream" in stage_name.lower() or "task" in stage_name.lower():
                    # This is the CRUD handler (read/create/update/delete)
                    operation = result.get("operation", "").lower()
                    display_name = f"{operation}_handler" if operation else "downstream_handler"
                else:
                    display_name = stage_name
                
                token_breakdown["by_stage"][display_name] = stage_tokens
        
        # BACKUP: Read from trace files if timing data is incomplete
        if token_breakdown["total_tokens"] == 0:
            reasoning_traces = traces_dir / "reasoning_traces"
            if reasoning_traces.exists():
                for trace_file in reasoning_traces.glob("*.json"):
                    try:
                        with open(trace_file, 'r') as f:
                            trace_data = json.load(f)
                        
                        # Check if this trace has token usage
                        if "token_usage" in trace_data:
                            usage = trace_data["token_usage"]
                            
                            # Determine stage name from filename
                            filename = trace_file.name
                            if filename.startswith("query_"):
                                stage_name = "xpath_generation"
                            elif filename.startswith("read_handler_"):
                                stage_name = "read_handler"
                            elif filename.startswith("create_handler_"):
                                stage_name = "create_handler"
                            elif filename.startswith("update_handler_"):
                                stage_name = "update_handler"
                            elif filename.startswith("delete_handler_"):
                                stage_name = "delete_handler"
                            elif filename.startswith("insertion_reasoner_"):
                                stage_name = "insertion_reasoner"
                            elif filename.startswith("node_reasoner_"):
                                stage_name = "node_reasoner"
                            else:
                                continue
                            
                            # Add to breakdown
                            token_breakdown["prompt_tokens"] += usage.get("prompt_tokens", 0)
                            token_breakdown["completion_tokens"] += usage.get("completion_tokens", 0)
                            token_breakdown["total_tokens"] += usage.get("total_tokens", 0)
                            token_breakdown["by_stage"][stage_name] = usage
                            
                    except (json.JSONDecodeError, IOError):
                        continue
        
        # Calculate cost
        token_breakdown["cost_usd"] = self._calculate_cost(token_breakdown)
        
        return token_breakdown
    
    def _run_session(
        self,
        query_list: List[str],
        session_name: str,
        session_index: int,
        use_versions: bool = False,
        tree_path: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run a session with multiple queries.
        
        Args:
            query_list: List of queries to execute in this session
            session_name: Display name for this session (used as folder name)
            session_index: 1-based session index (for display only)
            use_versions: Controls tree state behavior:
                - False: Each query runs independently on a fresh copy of the
                  original tree from storage. The tree is reset before each query.
                - True: CUD operations create new tree versions. Each consecutive
                  query runs on the updated tree, so changes accumulate.
            tree_path: Optional path to a custom starting tree. If provided,
                overrides the default tree from active_data. Can be absolute
                or relative to PROJECT_ROOT.
            
        Returns:
            Dict mapping query IDs to their results
        """
        # Sanitize session name for folder use
        safe_session_name = re.sub(r'[^\w\s-]', '', session_name)
        safe_session_name = re.sub(r'\s+', '_', safe_session_name).strip('_')
        if not safe_session_name:
            safe_session_name = f"Session_{session_index}"
        
        session_dir = self.experiment_dir / safe_session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        version_mode = "VERSIONED (cumulative)" if use_versions else "INDEPENDENT (fresh tree per query)"
        print(f"\n{'='*60}")
        print(f"Session {session_index} [{session_name}]: {len(query_list)} queries")
        print(f"  Mode: {version_mode}")
        print(f"  Scoring Method: {self.scoring_method}")
        print(f"  Scoring Model: {self.scoring_model}")
        if hasattr(self, 'scoring_api') and self.scoring_api:
            print(f"  Scoring API: {self.scoring_api}")
        print(f"{'='*60}")
        
        # Get source tree path: use custom tree_path if provided, otherwise default
        if tree_path:
            source_tree = Path(tree_path)
            if not source_tree.is_absolute():
                source_tree = PROJECT_ROOT / source_tree
            if not source_tree.exists():
                raise FileNotFoundError(f"Custom tree_path not found: {source_tree}")
            print(f"  Tree: {source_tree} (custom)")
        else:
            from pipeline_execution.semantic_xpath_execution import get_data_path
            source_tree = get_data_path(config=self.app_config)
            print(f"  Tree: {source_tree} (default)")
        
        # Copy tree to session directory
        session_tree = session_dir / "tree.xml"
        shutil.copy2(source_tree, session_tree)
        
        # When use_versions is True, we create one pipeline and let all queries
        # share the same tree state. CUD operations mutate tree.xml in-place,
        # so subsequent queries see the accumulated changes.
        # When use_versions is False, we create a fresh pipeline for each query
        # after resetting the tree to the original source.
        if use_versions:
            pipeline = SemanticXPathPipeline(tree_path=session_tree, config=self.app_config)
        
        session_results = {}
        
        for query_index, query in enumerate(query_list, start=1):
            query_name = self._sanitize_query_name(query, query_index)
            query_dir = session_dir / query_name
            query_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n  [{query_index}/{len(query_list)}] {query[:60]}...")
            
            if not use_versions:
                # Reset tree to original source before each query
                shutil.copy2(source_tree, session_tree)
                # Create a fresh pipeline so the executor re-parses the clean tree
                pipeline = SemanticXPathPipeline(tree_path=session_tree, config=self.app_config)
            
            # Set traces path for this query
            pipeline.set_traces_path(query_dir)
            
            # Execute query
            start_time = time.perf_counter()
            result = pipeline.process_request(query)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Save output tree to query folder when running independently
            # so each query's result tree can be verified individually.
            # When use_versions is true, the session-level tree.xml accumulates
            # all changes and serves as the final output.
            if not use_versions:
                shutil.copy2(session_tree, query_dir / "tree.xml")
            
            # Extract token usage and calculate cost from traces
            token_breakdown = self._extract_token_usage(result, query_dir)
            
            # Format and save query master log
            master_log = LogFormatter.format_query_master_log(
                user_query=query,
                pipeline_result=result,
                traces_dir=query_dir
            )
            master_log["execution_time_ms"] = round(execution_time, 2)
            master_log["token_usage"] = token_breakdown
            
            log_path = query_dir / "query_master_log.json"
            LogFormatter.save_query_master_log(master_log, log_path)
            
            # Extract top-k paths for experiment log
            top_k_paths = LogFormatter.extract_top_k_paths(result)
            
            # Store result for experiment log
            query_id = f"{safe_session_name}_{query_name}"
            session_results[query_id] = {
                "query": query,
                "operation": result.get("operation", "UNKNOWN"),
                "success": result.get("success", False),
                "top_k_nodes": top_k_paths,
                "execution_time_ms": round(execution_time, 2),
                "token_usage": token_breakdown
            }
            
            # Track for cost summary
            self._cost_tracker.append({
                "query_id": query_id,
                "query": query,
                "session": safe_session_name,
                "execution_time_ms": round(execution_time, 2),
                "token_usage": token_breakdown,
                "operation": result.get("operation", "UNKNOWN"),
                "success": result.get("success", False)
            })
            
            status = "SUCCESS" if result.get("success") else "FAILED"
            operation = result.get("operation", "UNKNOWN")
            cost_str = f"${token_breakdown['cost_usd']:.4f}"
            tokens_str = f"{token_breakdown['total_tokens']} tokens"
            print(f"      {status} | {operation} | {execution_time:.0f}ms | {tokens_str} | {cost_str}")
        
        return session_results
    
    def _build_cost_summary(self, total_experiment_time_ms: float) -> Dict[str, Any]:
        """
        Build detailed cost summary with per-task breakdown and totals.
        
        Args:
            total_experiment_time_ms: Total experiment execution time in milliseconds
            
        Returns:
            Dict with cost summary data
        """
        # Per-task breakdown
        tasks = []
        for entry in self._cost_tracker:
            tasks.append({
                "query_id": entry["query_id"],
                "session": entry["session"],
                "query": entry["query"],
                "operation": entry["operation"],
                "success": entry["success"],
                "latency_ms": entry["execution_time_ms"],
                "tokens": {
                    "prompt": entry["token_usage"]["prompt_tokens"],
                    "completion": entry["token_usage"]["completion_tokens"],
                    "total": entry["token_usage"]["total_tokens"]
                },
                "cost_usd": entry["token_usage"]["cost_usd"],
                "cost_breakdown_by_stage": entry["token_usage"]["by_stage"]
            })
        
        # Calculate totals
        total_latency = sum(t["latency_ms"] for t in tasks)
        total_prompt_tokens = sum(t["tokens"]["prompt"] for t in tasks)
        total_completion_tokens = sum(t["tokens"]["completion"] for t in tasks)
        total_tokens = sum(t["tokens"]["total"] for t in tasks)
        total_cost = sum(t["cost_usd"] for t in tasks)
        
        # Calculate averages
        num_tasks = len(tasks)
        avg_latency = total_latency / num_tasks if num_tasks > 0 else 0
        avg_tokens = total_tokens / num_tasks if num_tasks > 0 else 0
        avg_cost = total_cost / num_tasks if num_tasks > 0 else 0
        
        # Build summary
        return {
            "experiment_name": self.experiment_dir.name,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "summary": {
                "total_tasks": num_tasks,
                "total_latency_ms": round(total_latency, 2),
                "total_latency_seconds": round(total_latency / 1000, 2),
                "experiment_wall_time_ms": round(total_experiment_time_ms, 2),
                "experiment_wall_time_seconds": round(total_experiment_time_ms / 1000, 2),
                "total_tokens": {
                    "prompt": total_prompt_tokens,
                    "completion": total_completion_tokens,
                    "total": total_tokens
                },
                "total_cost_usd": round(total_cost, 4),
                "averages": {
                    "latency_ms": round(avg_latency, 2),
                    "tokens_per_task": round(avg_tokens, 2),
                    "cost_per_task_usd": round(avg_cost, 4)
                }
            },
            "tasks": tasks,
            "pricing_info": {
                "model": self.model,
                "rates": self._get_pricing_for_model()
            }
        }
    
    def _get_pricing_for_model(self) -> Dict[str, Any]:
        """Get pricing info for the current model."""
        model_lower = self.model.lower()
        for model_key in PRICING:
            if model_lower.startswith(model_key):
                return {
                    "prompt_per_1k_tokens": PRICING[model_key]["prompt"] * 1000,
                    "completion_per_1k_tokens": PRICING[model_key]["completion"] * 1000
                }
        # Default
        return {
            "prompt_per_1k_tokens": PRICING["gpt-4o"]["prompt"] * 1000,
            "completion_per_1k_tokens": PRICING["gpt-4o"]["completion"] * 1000
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full experiment.
        
        Returns:
            Experiment log dict
        """
        print(f"\n{'#'*60}")
        print(f"# Experiment: {self.experiment_name}")
        print(f"# Sessions: {len(self.sessions)}")
        print(f"# Scoring Method: {self.scoring_method}")
        print(f"# Scoring Model: {self.scoring_model}")
        if hasattr(self, 'scoring_api') and self.scoring_api:
            print(f"# Scoring API: {self.scoring_api}")
        print(f"{'#'*60}")
        
        # Setup experiment directory
        self.experiment_dir = self._setup_experiment_dir()
        print(f"\nOutput directory: {self.experiment_dir}")
        
        start_time = time.perf_counter()
        
        # Run each session
        for session_index, session in enumerate(self.sessions, start=1):
            session_results = self._run_session(
                session["queries"], session["name"], session_index,
                use_versions=session.get("use_versions", False),
                tree_path=session.get("tree_path")
            )
            self._session_results.update(session_results)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Build and save experiment log
        experiment_log = LogFormatter.format_experiment_log(
            experiment_name=self.experiment_dir.name,
            sessions_data=self._session_results
        )
        experiment_log["total_execution_time_ms"] = round(total_time, 2)
        
        # Calculate aggregate stats for experiment log
        total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) 
                          for r in self._session_results.values())
        total_cost = sum(r.get("token_usage", {}).get("cost_usd", 0) 
                        for r in self._session_results.values())
        
        experiment_log["total_tokens"] = total_tokens
        experiment_log["total_cost_usd"] = round(total_cost, 4)
        
        log_path = self.experiment_dir / "Experiment_Log.json"
        LogFormatter.save_experiment_log(experiment_log, log_path)
        
        # Build and save cost summary
        cost_summary = self._build_cost_summary(total_time)
        cost_summary_path = self.experiment_dir / "Cost_Summary.json"
        with open(cost_summary_path, "w", encoding="utf-8") as f:
            json.dump(cost_summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Experiment Complete!")
        print(f"{'='*60}")
        print(f"  Total sessions: {len(self.sessions)}")
        print(f"  Total queries: {len(self._session_results)}")
        
        success_count = sum(
            1 for r in self._session_results.values() if r.get("success")
        )
        print(f"  Successful: {success_count}/{len(self._session_results)}")
        print(f"  Total time: {total_time/1000:.1f}s")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"\nResults saved to: {self.experiment_dir}")
        print(f"  - Experiment_Log.json")
        print(f"  - Cost_Summary.json")
        
        return experiment_log


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run experiment with session-based query execution"
    )
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="experiment.yaml",
        help="Path to experiment config file (default: experiment.yaml)"
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.config)
    experiment_log = runner.run()
    
    # Print top-k results summary
    print("\n" + "="*60)
    print("Top-K Results Summary:")
    print("="*60)
    for query_id, data in experiment_log["results"].items():
        print(f"\n{query_id}:")
        print(f"  Query: {data['query'][:50]}...")
        print(f"  Top nodes:")
        for path in data.get("top_k_nodes", [])[:3]:
            print(f"    - {path}")


if __name__ == "__main__":
    main()
