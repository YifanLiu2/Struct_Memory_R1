"""
Flat-RAG Experiment Runner - Execute flat-rag baseline experiments.

Usage:
    python -m experiment.experiment_infra.flat-rag.flat_rag_runner --config experiment.yaml

Input Format (same as semantic_xpath and in_context runners):
    name: "experiment_name"
    queries:
      - name: "read_queries"
        use_versions: "false"
        queries:
          - "query 1"
          - "query 2"
    config:
      active_schema: "itinerary"
      active_data: "travel_toronto_10day"
      openai:
        model: "gpt-5-mini"

Output Structure (mirrors semantic_xpath):
    experiment/experiment_result/flat_rag/{experiment_name}/
    ├── Experiment_Log.json
    ├── Cost_Summary.json
    ├── experiment_config.yaml
    └── {session_name}/
        ├── tree.xml
        ├── 01_Query_{first_words}/
        │   ├── query_result.json
        │   ├── tree.xml                  (when use_versions=false)
        │   └── reasoning_traces/
        │       └── {handler}_*.json
        ├── 02_Query_{first_words}/
        │   └── ...
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Handle both direct execution and module import
try:
    from .flat_rag_pipeline import FlatRAGPipeline, FlatRAGResult
except ImportError:
    from flat_rag_pipeline import FlatRAGPipeline, FlatRAGResult

from pipeline_execution.semantic_xpath_util import get_data_path


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Token pricing (per token rates)
PRICING = {
    "gpt-4o": {"prompt": 0.0025 / 1000, "completion": 0.01 / 1000},
    "gpt-4o-mini": {"prompt": 0.00015 / 1000, "completion": 0.0006 / 1000},
    "gpt-5": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
    "gpt-5-mini": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
    "o1": {"prompt": 0.015 / 1000, "completion": 0.06 / 1000},
    "o3": {"prompt": 0.015 / 1000, "completion": 0.06 / 1000},
}


class FlatRAGRunner:
    """
    Runs flat-rag baseline experiments across multiple sessions.

    For each session:
    - Flattens tree leaf nodes into documents
    - Uses dense retrieval (TAS-B) to find top-k relevant nodes
    - Feeds retrieved nodes to downstream CRUD handlers
    - Logs results per-query in the same format as semantic_xpath
    """

    def __init__(self, config_path: str):
        """
        Initialize the experiment runner.

        Args:
            config_path: Path to experiment.yaml config file
        """
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        self.config_path = config_path
        self.config = self._load_config()

        self.experiment_name = self.config.get("name", "experiment")
        self.sessions = self._parse_sessions(self.config.get("queries", []))

        # Load embedded config
        embedded_config = self.config.get("config")
        if not embedded_config:
            raise ValueError(
                "Experiment config must have a 'config' section with all settings. "
                "See itinerary_experiment.yaml for the expected format."
            )
        self.app_config = embedded_config
        self.model = embedded_config.get("openai", {}).get("model", "gpt-4o")

        # Setup output directory
        self.base_output_dir = PROJECT_ROOT / "experiment" / "experiment_result" / "flat_rag"
        self.experiment_dir = None

        # Track results
        self._session_results: Dict[str, Dict[str, Any]] = {}
        self._cost_tracker: List[Dict[str, Any]] = []

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate experiment config."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        if "queries" not in config or not config["queries"]:
            raise ValueError("Experiment config must have non-empty 'queries' list")
        return config

    def _parse_sessions(self, queries_config: list) -> List[Dict[str, Any]]:
        """
        Parse queries config into named sessions.

        Supports:
          - Dict format: {name: "read_queries", queries: [...], use_versions: "false"}
          - List format: ["query1", "query2"] → auto-named Session_N
        """
        sessions = []
        unnamed_counter = 1

        for i, entry in enumerate(queries_config):
            if isinstance(entry, dict):
                name = entry.get("name")
                query_list = entry.get("queries", [])
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
                sessions.append({
                    "name": name, "queries": query_list,
                    "use_versions": use_versions,
                    "tree_path": tree_path,
                })
            elif isinstance(entry, list):
                if not entry:
                    raise ValueError(f"Query list at index {i} is empty")
                sessions.append({
                    "name": f"Session_{unnamed_counter}",
                    "queries": entry, "use_versions": False
                })
                unnamed_counter += 1
            else:
                raise ValueError(
                    f"Query entry at index {i} must be a list or dict."
                )

        return sessions

    def _resolve_source_tree_path(self, tree_path: Optional[str] = None) -> Path:
        """
        Resolve the source tree path for a session.

        Args:
            tree_path: Optional custom starting tree path. If provided, overrides
                default tree from active_data. Can be absolute or relative to PROJECT_ROOT.

        Returns:
            Path to an existing XML file.
        """
        if tree_path:
            source_tree = Path(tree_path)
            if not source_tree.is_absolute():
                source_tree = PROJECT_ROOT / source_tree
            if not source_tree.exists():
                raise FileNotFoundError(f"Custom tree_path not found: {source_tree}")
            print(f"  Tree: {source_tree} (custom)")
            return source_tree
        default_tree = get_data_path(config=self.app_config)
        print(f"  Tree: {default_tree} (default)")
        return default_tree

    def _setup_experiment_dir(self) -> Path:
        """Create experiment output directory (with dedup suffix)."""
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r'[^\w\s-]', '', self.experiment_name)
        safe_name = re.sub(r'\s+', '_', safe_name).strip('_') or "experiment"

        experiment_dir = self.base_output_dir / safe_name
        if experiment_dir.exists():
            counter = 2
            while (self.base_output_dir / f"{safe_name}_{counter}").exists():
                counter += 1
            experiment_dir = self.base_output_dir / f"{safe_name}_{counter}"

        experiment_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.config_path, experiment_dir / "experiment_config.yaml")
        return experiment_dir

    def _sanitize_query_name(self, query: str, index: int) -> str:
        """Create sanitized folder name from query text."""
        words = query.split()[:4]
        name = "_".join(words)
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"\s+", "_", name)
        name = name[:30]
        return f"{index:02d}_Query_{name}" if name else f"{index:02d}_Query"

    def _calculate_cost(self, token_usage: Dict[str, Any]) -> float:
        """Calculate cost in USD for the given token usage."""
        pricing = None
        model_lower = self.model.lower()
        for model_key in PRICING:
            if model_lower.startswith(model_key):
                pricing = PRICING[model_key]
                break
        if not pricing:
            pricing = PRICING["gpt-4o"]

        prompt_cost = token_usage.get("prompt_tokens", 0) * pricing["prompt"]
        completion_cost = token_usage.get("completion_tokens", 0) * pricing["completion"]
        return prompt_cost + completion_cost

    def _get_pricing_for_model(self) -> Dict[str, Any]:
        """Get pricing info for the current model."""
        model_lower = self.model.lower()
        for model_key in PRICING:
            if model_lower.startswith(model_key):
                return {
                    "prompt_per_1k_tokens": PRICING[model_key]["prompt"] * 1000,
                    "completion_per_1k_tokens": PRICING[model_key]["completion"] * 1000
                }
        return {
            "prompt_per_1k_tokens": PRICING["gpt-4o"]["prompt"] * 1000,
            "completion_per_1k_tokens": PRICING["gpt-4o"]["completion"] * 1000
        }

    # ----------------------------------------------------------------
    # Flattened document saving
    # ----------------------------------------------------------------

    def _save_flat_docs(self, output_dir: Path, pipeline: FlatRAGPipeline):
        """
        Save the flattened leaf-node documents to the results directory.

        Creates a flat_documents.json file containing all flattened documents
        with their tree paths and document text used for embedding.
        """
        flat_docs = pipeline.get_flat_documents()
        output_path = output_dir / "flat_documents.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_documents": len(flat_docs),
                "documents": flat_docs
            }, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(flat_docs)} flattened documents to {output_path.name}")

    # ----------------------------------------------------------------
    # Per-query result saving
    # ----------------------------------------------------------------

    def _save_query_result(
        self,
        query_dir: Path,
        user_query: str,
        result: FlatRAGResult,
        cost_usd: float
    ):
        """
        Save per-query result files: query_result.json + reasoning_traces/.

        Creates:
            query_dir/
            ├── query_result.json
            └── reasoning_traces/
                └── (CRUD handler traces are saved by the handlers themselves)
        """
        query_dir.mkdir(parents=True, exist_ok=True)

        # Build query result
        handler_output = result.handler_result or {}

        query_result = {
            "user_query": user_query,
            "operation": result.operation,
            "success": result.success,
            "version_used": result.version_used,
            "retrieved_documents": result.retrieved_docs,
            "retrieval_time_ms": round(result.retrieval_time_ms, 2),
            "downstream_task_result": handler_output,
            "execution_time_ms": round(result.execution_time_ms, 2),
            "token_usage": result.token_usage,
            "cost_usd": round(cost_usd, 5),
            "error": result.error
        }

        with open(query_dir / "query_result.json", "w", encoding="utf-8") as f:
            json.dump(query_result, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------------
    # Session execution
    # ----------------------------------------------------------------

    def _run_session(
        self,
        query_list: List[str],
        session_name: str,
        session_index: int,
        use_versions: bool = False,
        tree_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run a session with multiple queries.

        Args:
            query_list: Queries to execute
            session_name: Display/folder name
            session_index: 1-based index
            use_versions: Whether CUD ops create new versions
        """
        safe_session_name = re.sub(r'[^\w\s-]', '', session_name)
        safe_session_name = re.sub(r'\s+', '_', safe_session_name).strip('_')
        if not safe_session_name:
            safe_session_name = f"Session_{session_index}"

        session_dir = self.experiment_dir / safe_session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        version_mode = (
            "VERSIONED (cumulative)" if use_versions
            else "INDEPENDENT (fresh tree per query)"
        )
        print(f"\n{'='*60}")
        print(f"Session {session_index} [{session_name}]: {len(query_list)} queries")
        print(f"  Mode: {version_mode}")
        print(f"{'='*60}")

        # Get source tree path: use custom tree_path if provided, otherwise default
        source_tree = self._resolve_source_tree_path(tree_path)
        session_tree = session_dir / "tree.xml"
        shutil.copy2(source_tree, session_tree)

        # Create pipeline (only once if use_versions=True)
        if use_versions:
            pipeline = FlatRAGPipeline(
                tree_path=session_tree, config=self.app_config
            )
            # Save flattened documents to results
            self._save_flat_docs(session_dir, pipeline)

        session_results = {}

        for query_index, query in enumerate(query_list, start=1):
            query_name = self._sanitize_query_name(query, query_index)
            query_dir = session_dir / query_name
            query_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  [{query_index}/{len(query_list)}] {query[:60]}...")

            if not use_versions:
                # Fresh tree per query
                shutil.copy2(source_tree, session_tree)
                pipeline = FlatRAGPipeline(
                    tree_path=session_tree, config=self.app_config
                )
                # Save flattened documents once (same source tree for all queries)
                if query_index == 1:
                    self._save_flat_docs(session_dir, pipeline)

            # Set traces path so CRUD handlers save to query dir
            pipeline.set_traces_path(query_dir)

            # Execute query
            start_time = time.perf_counter()
            result = pipeline.process_request(query)
            execution_time = (time.perf_counter() - start_time) * 1000

            # Save output tree to query folder (independent mode)
            if not use_versions:
                shutil.copy2(session_tree, query_dir / "tree.xml")

            # Calculate cost
            cost_usd = self._calculate_cost(result.token_usage)
            result.token_usage["cost_usd"] = round(cost_usd, 5)

            # Save per-query results
            self._save_query_result(query_dir, query, result, cost_usd)

            # Extract top-k paths for experiment log
            top_k_paths = [
                doc.get("tree_path", "") for doc in result.retrieved_docs[:5]
            ]

            # Store for experiment log
            query_id = f"{safe_session_name}_{query_name}"
            session_results[query_id] = {
                "query": query,
                "operation": result.operation,
                "success": result.success,
                "top_k_nodes": top_k_paths,
                "execution_time_ms": round(execution_time, 2),
                "token_usage": result.token_usage
            }

            # Track for cost summary
            self._cost_tracker.append({
                "query_id": query_id,
                "query": query,
                "session": safe_session_name,
                "execution_time_ms": round(execution_time, 2),
                "token_usage": result.token_usage,
                "operation": result.operation,
                "success": result.success
            })

            # Print status
            status = "SUCCESS" if result.success else "FAILED"
            op = result.operation
            tokens_str = f"{result.token_usage.get('total_tokens', 0)} tokens"
            print(
                f"      {status} | {op} | Version {result.version_used} "
                f"| {execution_time:.0f}ms | {tokens_str} | ${cost_usd:.4f}"
            )
            for doc in result.retrieved_docs[:3]:
                print(f"        → {doc.get('tree_path', '')} ({doc.get('score', 0):.3f})")

        return session_results

    # ----------------------------------------------------------------
    # Cost summary
    # ----------------------------------------------------------------

    def _build_cost_summary(self, total_experiment_time_ms: float) -> Dict[str, Any]:
        """Build detailed cost summary."""
        tasks = []
        for entry in self._cost_tracker:
            tu = entry.get("token_usage", {})
            tasks.append({
                "query_id": entry["query_id"],
                "session": entry["session"],
                "query": entry["query"],
                "operation": entry["operation"],
                "success": entry["success"],
                "latency_ms": entry["execution_time_ms"],
                "tokens": {
                    "prompt": tu.get("prompt_tokens", 0),
                    "completion": tu.get("completion_tokens", 0),
                    "total": tu.get("total_tokens", 0)
                },
                "cost_usd": tu.get("cost_usd", 0),
                "cost_breakdown_by_stage": tu.get("by_stage", {})
            })

        total_latency = sum(t["latency_ms"] for t in tasks)
        total_prompt = sum(t["tokens"]["prompt"] for t in tasks)
        total_completion = sum(t["tokens"]["completion"] for t in tasks)
        total_tokens = sum(t["tokens"]["total"] for t in tasks)
        total_cost = sum(t["cost_usd"] for t in tasks)
        num_tasks = len(tasks)

        return {
            "experiment_name": self.experiment_dir.name,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "method": "flat-rag",
            "summary": {
                "total_tasks": num_tasks,
                "total_latency_ms": round(total_latency, 2),
                "total_latency_seconds": round(total_latency / 1000, 2),
                "experiment_wall_time_ms": round(total_experiment_time_ms, 2),
                "experiment_wall_time_seconds": round(total_experiment_time_ms / 1000, 2),
                "total_tokens": {
                    "prompt": total_prompt,
                    "completion": total_completion,
                    "total": total_tokens
                },
                "total_cost_usd": round(total_cost, 4),
                "averages": {
                    "latency_ms": round(total_latency / num_tasks, 2) if num_tasks else 0,
                    "tokens_per_task": round(total_tokens / num_tasks, 2) if num_tasks else 0,
                    "cost_per_task_usd": round(total_cost / num_tasks, 4) if num_tasks else 0
                }
            },
            "tasks": tasks,
            "pricing_info": {
                "model": self.model,
                "rates": self._get_pricing_for_model()
            }
        }

    # ----------------------------------------------------------------
    # Main run
    # ----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the full experiment."""
        print(f"\n{'#'*60}")
        print(f"# Flat-RAG Experiment: {self.experiment_name}")
        print(f"# Sessions: {len(self.sessions)}")
        print(f"{'#'*60}")

        self.experiment_dir = self._setup_experiment_dir()
        print(f"\nOutput directory: {self.experiment_dir}")

        start_time = time.perf_counter()

        for session_index, session in enumerate(self.sessions, start=1):
            session_results = self._run_session(
                session["queries"], session["name"], session_index,
                use_versions=session.get("use_versions", False),
                tree_path=session.get("tree_path"),
            )
            self._session_results.update(session_results)

        total_time = (time.perf_counter() - start_time) * 1000

        # Save experiment log
        experiment_log = {
            "experiment_name": self.experiment_dir.name,
            "method": "flat-rag",
            "timestamp": datetime.now().isoformat(),
            "total_sessions": len(self.sessions),
            "total_queries": len(self._session_results),
            "results": self._session_results,
            "total_execution_time_ms": round(total_time, 2)
        }

        total_tokens = sum(
            r.get("token_usage", {}).get("total_tokens", 0)
            for r in self._session_results.values()
        )
        total_cost = sum(
            r.get("token_usage", {}).get("cost_usd", 0)
            for r in self._session_results.values()
        )
        experiment_log["total_tokens"] = total_tokens
        experiment_log["total_cost_usd"] = round(total_cost, 4)

        log_path = self.experiment_dir / "Experiment_Log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(experiment_log, f, indent=2, ensure_ascii=False)

        # Save cost summary
        cost_summary = self._build_cost_summary(total_time)
        with open(self.experiment_dir / "Cost_Summary.json", "w", encoding="utf-8") as f:
            json.dump(cost_summary, f, indent=2, ensure_ascii=False)

        # Print summary
        success_count = sum(
            1 for r in self._session_results.values() if r.get("success")
        )
        print(f"\n{'='*60}")
        print(f"Flat-RAG Experiment Complete!")
        print(f"{'='*60}")
        print(f"  Total sessions: {len(self.sessions)}")
        print(f"  Total queries: {len(self._session_results)}")
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
        description="Run flat-rag baseline experiment"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="experiment.yaml",
        help="Path to experiment config file (default: experiment.yaml)"
    )

    args = parser.parse_args()

    runner = FlatRAGRunner(args.config)
    experiment_log = runner.run()

    # Print results summary
    print("\n" + "=" * 60)
    print("Top-K Results Summary:")
    print("=" * 60)
    for query_id, data in list(experiment_log.get("results", {}).items())[:5]:
        print(f"\n{query_id}:")
        print(f"  Query: {data.get('query', '')[:50]}...")
        print(f"  Operation: {data.get('operation', 'UNKNOWN')}")
        print(f"  Top nodes:")
        for path in data.get("top_k_nodes", [])[:3]:
            print(f"    - {path}")


if __name__ == "__main__":
    main()
