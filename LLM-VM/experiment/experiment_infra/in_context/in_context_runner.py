"""
In-Context Experiment Runner - Execute in-context evaluation experiments.

Usage:
    python experiment/experiment_infra/in_context/in_context_runner.py --config experiment.yaml
    # or
    python -m experiment.experiment_infra.in_context.in_context_runner --config experiment.yaml

Input Format (same as semantic_xpath_experiment_runner):
    name: "experiment_name"
    queries:
      # --- Single-turn sessions (independent queries) ---
      - name: "read_queries"
        use_versions: "false"
        queries:
          - "query 1"
          - "query 2"

      # --- Multi-turn sessions (sequential queries on evolving tree) ---
      # Each named group whose name contains "multi-turn" or "multi_turn"
      # with use_versions: "true" is treated as an independent multi-turn run.
      # All queries in the flat list are processed sequentially on the same
      # evolving tree.  Different named groups start from the original tree.
      - name: 'multi-turn_0'
        use_versions: "true"
        queries:
          - "turn 1 query"
          - "turn 2 query"
          - "turn 3 query"
      - name: 'multi-turn_1'
        use_versions: "true"
        queries:
          - "turn 1 query"   # independent run, fresh tree
          - "turn 2 query"

    config:
      active_schema: "itinerary"
      active_data: "travel_toronto_10day"
      openai:
        model: "gpt-5-mini"
        ...

Output Structure:
    experiment/experiment_result/in_context/{experiment_name}/
    ├── Experiment_Log.json
    ├── Cost_Summary.json
    ├── experiment_config.yaml
    ├── {session_name}/                      # Single-turn sessions
    │   ├── tree.xml
    │   ├── session_result.json
    │   ├── 01_Query_{first_words}/
    │   │   ├── query_result.json
    │   │   ├── tree.xml
    │   │   └── reasoning_traces/
    │   │       └── llm_response_{timestamp}.json
    │   └── ...
    └── multi-turn/                          # Multi-turn parent directory
        ├── multi-turn_0/                    # Independent multi-turn run
        │   ├── tree.xml                     # Evolving tree for this run
        │   ├── session_result.json          # Summary for this conversation
        │   ├── 01_Turn_{first_words}/
        │   │   ├── query_result.json
        │   │   ├── tree.xml                 # Tree snapshot before this turn
        │   │   └── reasoning_traces/
        │   │       └── llm_response_{timestamp}.json
        │   ├── 02_Turn_{first_words}/
        │   │   └── ...
        │   └── ...
        ├── multi-turn_1/                    # Another independent run (fresh tree)
        │   ├── tree.xml
        │   ├── session_result.json
        │   ├── 01_Turn_{first_words}/
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
import xml.etree.ElementTree as ET
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Handle both direct execution and module import
try:
    from .in_context_pipeline import InContextPipeline, InContextResult
    from .result_logger import InContextResultLogger, SessionResult, QueryResult
except ImportError:
    from in_context_pipeline import InContextPipeline, InContextResult
    from result_logger import InContextResultLogger, SessionResult, QueryResult
from pipeline_execution.semantic_xpath_util import get_data_path


# Project root directory
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


class InContextRunner:
    """
    Runs in-context evaluation experiments across multiple sessions.
    
    For each session:
    - Single-turn: Copy fresh tree, process single query, log result
    - Multi-turn: Copy tree once, process queries sequentially with
                 accumulated versions, LLM sees full tree each turn
    
    The LLM autonomously decides which version to operate on based
    on the user's query.
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
                "See itinerary_experiment.yaml for the expected format."
            )
        self.app_config = embedded_config
        self.model = embedded_config.get("openai", {}).get("model", "gpt-4o")
        
        # Setup output directory
        self.base_output_dir = PROJECT_ROOT / "experiment" / "experiment_result" / "in_context"
        self.experiment_dir = None  # Set during run()
        
        # Result logger
        self.result_logger = None  # Set during run()
        
        # Cost tracking
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
                sessions.append(
                    {
                        "name": name,
                        "queries": query_list,
                        "use_versions": use_versions,
                        "tree_path": tree_path,
                    }
                )
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
        default_tree = self._get_source_tree_path()
        print(f"  Tree: {default_tree} (default)")
        return default_tree
    
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
    
    def _get_source_tree_path(self) -> Path:
        """Get the source tree path from embedded experiment config."""
        return get_data_path(config=self.app_config)
    
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
    
    def _sanitize_query_name(self, query: str, index: int) -> str:
        """
        Create a sanitized folder name from query text, prefixed with index for ordering.
        
        Matches the semantic_xpath runner convention:
            01_Query_For_all_days_that
            02_Query_The_workshop_on_Day
        """
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
    
    def _is_multi_turn_session(self, session_name: str) -> bool:
        """
        Check if a session name indicates a multi-turn conversation.
        
        A session is treated as multi-turn if its name contains "multi-turn"
        or "multi_turn" (case-insensitive).
        
        Args:
            session_name: Name of the session from the YAML config
            
        Returns:
            True if the session is a multi-turn conversation
        """
        normalized = session_name.lower().replace("-", "_")
        return "multi_turn" in normalized
    
    def _run_multi_turn_session(
        self,
        turn_queries: List[str],
        session_name: str,
        multi_turn_parent_dir: Path,
        session_index: int,
        tree_path: Optional[str] = None,
    ) -> SessionResult:
        """
        Run a complete independent multi-turn session.
        
        Each multi-turn session:
        - Starts from a fresh copy of the original source tree
        - Processes ALL queries sequentially as turns on the same evolving tree
        - CUD operations create new versions; subsequent turns see all changes
        - Stores results under multi-turn/<session_name>/
        
        Args:
            turn_queries: Flat list of queries to process sequentially
            session_name: Name of this multi-turn group (e.g., "multi-turn_0")
            multi_turn_parent_dir: Parent directory (multi-turn/)
            session_index: 1-based index for display
            
        Returns:
            SessionResult with all turn results
        """
        # Sanitize name for folder use
        safe_name = re.sub(r'[^\w\s-]', '', session_name)
        safe_name = re.sub(r'\s+', '_', safe_name).strip('_')
        if not safe_name:
            safe_name = f"multi_turn_{session_index}"
        
        # Create session directory under multi-turn/
        session_dir = multi_turn_parent_dir / safe_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Multi-Turn Session {session_index} [{session_name}]: {len(turn_queries)} turns")
        print(f"  Mode: MULTI-TURN (sequential, versioned)")
        print(f"  Output: {session_dir.relative_to(self.experiment_dir)}")
        print(f"{'='*60}")
        
        # Copy fresh tree from original source (independent from other sessions)
        source_tree = self._resolve_source_tree_path(tree_path)
        session_tree_path = session_dir / "tree.xml"
        shutil.copy2(source_tree, session_tree_path)
        
        # Create fresh pipeline for this independent run
        pipeline = InContextPipeline(tree_path=session_tree_path, config=self.app_config)
        
        session_result = SessionResult(session_id=safe_name)
        conversation_history: List[Dict[str, str]] = []
        multi_turn_results: List[Dict[str, Any]] = []
        current_tree_version = 1
        
        for turn_index, query in enumerate(turn_queries, start=1):
            turn_name = self._sanitize_query_name(query, turn_index).replace("Query", "Turn")
            turn_dir = session_dir / turn_name
            turn_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n  [Turn {turn_index}/{len(turn_queries)}] {query[:60]}...")
            
            # Read current tree state (with all accumulated versions)
            with open(session_tree_path, "r") as f:
                tree_xml = f.read()
            
            # Save current tree snapshot for this turn (before processing)
            shutil.copy2(session_tree_path, turn_dir / "tree.xml")
            
            # Process query - LLM sees full tree and decides which version to operate on
            result = pipeline.process_request(
                query, tree_xml,
                conversation_history=conversation_history if conversation_history else None
            )
            
            # Build user message for history tracking
            user_message_for_history = f"## Current Tree State\n\n{tree_xml}\n\n## User Request\n\n{query}"
            
            # Append this turn to conversation history for next turn
            conversation_history.append({
                "role": "user",
                "content": user_message_for_history
            })
            conversation_history.append({
                "role": "assistant",
                "content": result.raw_response
            })
            
            # Format and add to session result
            query_result = InContextResultLogger.format_query_result(
                turn_index, query, result
            )
            session_result.add_query_result(query_result)
            
            # For CUD operations, apply modifications and save tree
            if result.operation in ("CREATE", "UPDATE", "DELETE") and result.success:
                tree = ET.parse(session_tree_path)
                success, tree = pipeline.apply_modifications(result, tree, query)
                if success:
                    pipeline.save_tree(tree, session_tree_path)
                    current_tree_version = pipeline.version_manager.get_version_count(tree)
                    session_result.final_tree_version = current_tree_version
            
            # Calculate cost
            token_usage = result.token_usage if result.token_usage else {}
            cost_usd = self._calculate_cost(token_usage)
            
            # Save per-turn result files (query_result.json + reasoning_traces/)
            self.result_logger.save_query_result(
                query_dir=turn_dir,
                user_query=query,
                pipeline_result=result,
                input_xml=tree_xml,
                cost_usd=cost_usd
            )
            
            # Collect turn result for the multi-turn summary
            multi_turn_results.append({
                "turn_index": turn_index,
                "user_query": query,
                "operation": result.operation,
                "success": result.success,
                "version_used": result.version_used,
                "reasoning": result.reasoning,
                "selected_nodes": result.selected_nodes,
                "related_nodes": result.related_nodes,
                "execution_time_ms": round(result.execution_time_ms, 2),
                "token_usage": token_usage,
                "cost_usd": round(cost_usd, 5),
                "error": result.error
            })
            
            # Track for cost summary
            query_id = f"{safe_name}_{turn_name}"
            self._cost_tracker.append({
                "query_id": query_id,
                "session": safe_name,
                "query": query,
                "operation": result.operation,
                "success": result.success,
                "latency_ms": round(result.execution_time_ms, 2),
                "token_usage": token_usage,
                "cost_usd": round(cost_usd, 5)
            })
            
            # Print status
            status = "SUCCESS" if result.success else "FAILED"
            total_tokens = token_usage.get("total_tokens", 0)
            print(
                f"      {status} | {result.operation} | Version {result.version_used} "
                f"| {result.execution_time_ms:.0f}ms | {total_tokens} tokens | ${cost_usd:.4f}"
            )
            if result.selected_nodes:
                for path in result.selected_nodes[:5]:
                    print(f"        → {path}")
        
        # Save per-session summary
        self.result_logger.save_multi_turn_summary(
            multi_turn_dir=session_dir,
            multi_turn_id=safe_name,
            turn_results=multi_turn_results,
            final_tree_version=current_tree_version,
        )
        
        return session_result
    
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
            token_usage = entry.get("token_usage", {})
            tasks.append({
                "query_id": entry["query_id"],
                "session": entry["session"],
                "query": entry["query"],
                "operation": entry["operation"],
                "success": entry["success"],
                "latency_ms": entry["latency_ms"],
                "tokens": {
                    "prompt": token_usage.get("prompt_tokens", 0),
                    "completion": token_usage.get("completion_tokens", 0),
                    "total": token_usage.get("total_tokens", 0)
                },
                "cost_usd": entry["cost_usd"]
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
    
    def _run_multi_turn(
        self,
        turn_queries: List[str],
        session_dir: Path,
        session_tree_path: Path,
        pipeline: InContextPipeline,
        session_result: SessionResult,
        safe_session_name: str,
        multi_turn_index: int = 1
    ):
        """
        Run a multi-turn conversation: queries processed sequentially with
        previous LLM responses appended as conversation history.
        
        Each multi-turn list gets its own folder with a per-conversation
        session_result.json summary.
        
        Args:
            turn_queries: Ordered list of queries forming a multi-turn conversation
            session_dir: Session output directory
            session_tree_path: Path to session tree file
            pipeline: InContextPipeline instance
            session_result: SessionResult to accumulate results into
            safe_session_name: Sanitized session name for IDs
            multi_turn_index: 1-based index when multiple multi-turn convos exist
        """
        # Each multi-turn conversation gets its own dedicated folder
        multi_turn_id = f"multi_turn_{multi_turn_index}"
        multi_turn_dir = session_dir / multi_turn_id
        multi_turn_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  --- Multi-Turn Conversation {multi_turn_index} ({len(turn_queries)} turns) ---")
        
        # Conversation history: list of {"role": "user"|"assistant", "content": "..."}
        conversation_history: List[Dict[str, str]] = []
        
        # Collect per-turn results for the multi-turn summary
        multi_turn_results: List[Dict[str, Any]] = []
        current_tree_version = 1
        
        for turn_index, query in enumerate(turn_queries, start=1):
            turn_name = self._sanitize_query_name(query, turn_index).replace("Query", "Turn")
            turn_dir = multi_turn_dir / turn_name
            turn_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n  [Turn {turn_index}/{len(turn_queries)}] {query[:60]}...")
            
            # Read current tree state
            with open(session_tree_path, "r") as f:
                tree_xml = f.read()
            
            # Process query with conversation history from previous turns
            result = pipeline.process_request(
                query, tree_xml,
                conversation_history=conversation_history if conversation_history else None
            )
            
            # Build the user message that was sent (for history tracking)
            user_message_for_history = f"## Current Tree State\n\n{tree_xml}\n\n## User Request\n\n{query}"
            
            # Append this turn to conversation history for next turn
            conversation_history.append({
                "role": "user",
                "content": user_message_for_history
            })
            conversation_history.append({
                "role": "assistant",
                "content": result.raw_response
            })
            
            # Format and add to session result
            global_index = (multi_turn_index - 1) * len(turn_queries) + turn_index
            query_result = InContextResultLogger.format_query_result(
                global_index, query, result
            )
            session_result.add_query_result(query_result)
            
            # For CUD operations, apply modifications and save tree
            if result.operation in ("CREATE", "UPDATE", "DELETE") and result.success:
                tree = ET.parse(session_tree_path)
                success, tree = pipeline.apply_modifications(result, tree, query)
                if success:
                    pipeline.save_tree(tree, session_tree_path)
                    current_tree_version = pipeline.version_manager.get_version_count(tree)
                    session_result.final_tree_version = current_tree_version
            
            # Calculate cost
            token_usage = result.token_usage if result.token_usage else {}
            cost_usd = self._calculate_cost(token_usage)
            
            # Save per-turn result files
            self.result_logger.save_query_result(
                query_dir=turn_dir,
                user_query=query,
                pipeline_result=result,
                input_xml=tree_xml,
                cost_usd=cost_usd
            )
            
            # Collect turn result for the multi-turn summary
            multi_turn_results.append({
                "turn_index": turn_index,
                "user_query": query,
                "operation": result.operation,
                "success": result.success,
                "version_used": result.version_used,
                "reasoning": result.reasoning,
                "selected_nodes": result.selected_nodes,
                "related_nodes": result.related_nodes,
                "execution_time_ms": round(result.execution_time_ms, 2),
                "token_usage": token_usage,
                "cost_usd": round(cost_usd, 5),
                "error": result.error
            })
            
            # Track for cost summary
            query_id = f"{safe_session_name}_{multi_turn_id}_{turn_name}"
            self._cost_tracker.append({
                "query_id": query_id,
                "session": safe_session_name,
                "query": query,
                "operation": result.operation,
                "success": result.success,
                "latency_ms": round(result.execution_time_ms, 2),
                "token_usage": token_usage,
                "cost_usd": round(cost_usd, 5)
            })
            
            # Print status
            status = "SUCCESS" if result.success else "FAILED"
            total_tokens = token_usage.get("total_tokens", 0)
            print(
                f"      {status} | {result.operation} | Version {result.version_used} "
                f"| {result.execution_time_ms:.0f}ms | {total_tokens} tokens | ${cost_usd:.4f}"
            )
            if result.selected_nodes:
                for path in result.selected_nodes[:5]:
                    print(f"        → {path}")
        
        # Save per-multi-turn session summary
        self.result_logger.save_multi_turn_summary(
            multi_turn_dir=multi_turn_dir,
            multi_turn_id=multi_turn_id,
            turn_results=multi_turn_results,
            final_tree_version=current_tree_version,
        )
    
    def _run_session(
        self,
        query_list: List,
        session_name: str,
        session_index: int,
        use_versions: bool = False,
        tree_path: Optional[str] = None,
    ) -> SessionResult:
        """
        Run a session with one or more queries.
        
        Args:
            query_list: List of queries. Each entry can be:
                - A string: single query
                - A list of strings: multi-turn conversation (when use_versions=True)
            session_name: Display name for this session (used as folder name)
            session_index: 1-based session index (for display only)
            use_versions: Controls tree state behavior:
                - False: Each query runs independently on a fresh copy of the
                  original tree from storage. The tree is reset before each query.
                - True: CUD operations create new versions. Each consecutive
                  query runs on the updated tree, so changes accumulate.
                  If a query entry is a list, it is treated as a multi-turn
                  conversation where each subsequent turn receives the previous
                  LLM response as conversation context.
            
        Returns:
            SessionResult with all query results
        """
        # Sanitize session name for folder use
        safe_session_name = re.sub(r'[^\w\s-]', '', session_name)
        safe_session_name = re.sub(r'\s+', '_', safe_session_name).strip('_')
        if not safe_session_name:
            safe_session_name = f"Session_{session_index}"
        
        session_dir = self.result_logger.create_session_dir(safe_session_name)
        
        version_mode = "VERSIONED (cumulative)" if use_versions else "INDEPENDENT (fresh tree per query)"
        print(f"\n{'='*60}")
        print(f"Session {session_index} [{session_name}]: {len(query_list)} queries")
        print(f"  Mode: {version_mode}")
        print(f"{'='*60}")
        
        # Get source tree path: use custom tree_path if provided, otherwise default
        source_tree = self._resolve_source_tree_path(tree_path)
        
        # Copy tree to session directory
        session_tree_path = session_dir / "tree.xml"
        shutil.copy2(source_tree, session_tree_path)
        
        # Create pipeline with experiment config
        pipeline = InContextPipeline(tree_path=session_tree_path, config=self.app_config)
        
        session_result = SessionResult(session_id=safe_session_name)
        
        single_query_index = 0  # Track index for single queries
        multi_turn_index = 0    # Track index for multi-turn conversations
        
        for entry in query_list:
            # Check if this entry is a multi-turn conversation (list of queries)
            if isinstance(entry, list) and use_versions:
                multi_turn_index += 1
                self._run_multi_turn(
                    turn_queries=entry,
                    session_dir=session_dir,
                    session_tree_path=session_tree_path,
                    pipeline=pipeline,
                    session_result=session_result,
                    safe_session_name=safe_session_name,
                    multi_turn_index=multi_turn_index
                )
                continue
            
            # Single query (string)
            query = entry
            single_query_index += 1
            
            # Create per-query directory
            query_name = self._sanitize_query_name(query, single_query_index)
            query_dir = session_dir / query_name
            query_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n  [{single_query_index}] {query[:60]}...")
            
            if not use_versions:
                # Reset tree to original source before each query
                shutil.copy2(source_tree, session_tree_path)
                # Reload pipeline tree so it sees the fresh tree
                pipeline.reload_tree()
            
            # Read current tree state (with all versions for multi-turn)
            with open(session_tree_path, "r") as f:
                tree_xml = f.read()
            
            # Process query through pipeline
            # LLM sees full tree and decides which version to operate on
            result = pipeline.process_request(query, tree_xml)
            
            # Format and add query result
            query_result = InContextResultLogger.format_query_result(
                single_query_index, query, result
            )
            session_result.add_query_result(query_result)
            
            # For CUD operations with use_versions, apply modifications and save tree
            if result.operation in ("CREATE", "UPDATE", "DELETE") and result.success and use_versions:
                # Reload tree and apply modifications
                tree = ET.parse(session_tree_path)
                success, tree = pipeline.apply_modifications(result, tree, query)
                
                if success:
                    # Save modified tree (creates new version)
                    pipeline.save_tree(tree, session_tree_path)
                    session_result.final_tree_version = pipeline.version_manager.get_version_count(tree)
            
            # Calculate cost from token usage
            token_usage = result.token_usage if result.token_usage else {}
            cost_usd = self._calculate_cost(token_usage)
            
            # Save per-query result files (query_result.json + reasoning_traces/)
            self.result_logger.save_query_result(
                query_dir=query_dir,
                user_query=query,
                pipeline_result=result,
                input_xml=tree_xml,
                cost_usd=cost_usd
            )
            
            # Save output tree to query folder when running independently
            # so each query's result tree can be verified individually.
            # When use_versions is true, the session-level tree.xml accumulates
            # all changes and serves as the final output.
            if not use_versions:
                shutil.copy2(session_tree_path, query_dir / "tree.xml")
            
            # Track for cost summary
            query_id = f"{safe_session_name}_{query_name}"
            self._cost_tracker.append({
                "query_id": query_id,
                "session": safe_session_name,
                "query": query,
                "operation": result.operation,
                "success": result.success,
                "latency_ms": round(result.execution_time_ms, 2),
                "token_usage": token_usage,
                "cost_usd": round(cost_usd, 5)
            })
            
            # Print status with cost info
            status = "SUCCESS" if result.success else "FAILED"
            operation = result.operation
            total_tokens = token_usage.get("total_tokens", 0)
            print(f"      {status} | {operation} | Version {result.version_used} | {result.execution_time_ms:.0f}ms | {total_tokens} tokens | ${cost_usd:.4f}")
            
            if result.selected_nodes:
                for path in result.selected_nodes[:5]:
                    print(f"        → {path}")
        
        # Save session result file
        self.result_logger.save_session_result(session_result, session_dir)
        
        return session_result
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full experiment.
        
        Separates sessions into two categories:
        1. Regular sessions (single-turn or versioned sequential queries)
        2. Multi-turn sessions (name contains "multi-turn"/"multi_turn" with
           use_versions=true) — each is an independent run with fresh tree,
           results stored under multi-turn/<session_name>/
        
        Returns:
            Experiment log dict
        """
        print(f"\n{'#'*60}")
        print(f"# In-Context Experiment: {self.experiment_name}")
        print(f"# Sessions: {len(self.sessions)}")
        print(f"{'#'*60}")
        
        # Setup experiment directory
        self.experiment_dir = self._setup_experiment_dir()
        self.result_logger = InContextResultLogger(self.experiment_dir)
        
        print(f"\nOutput directory: {self.experiment_dir}")
        
        # Separate multi-turn sessions from regular sessions
        regular_sessions = []
        multi_turn_sessions = []
        
        for session in self.sessions:
            if self._is_multi_turn_session(session["name"]) and session.get("use_versions", False):
                multi_turn_sessions.append(session)
            else:
                regular_sessions.append(session)
        
        if regular_sessions:
            print(f"\n  Regular sessions: {len(regular_sessions)}")
        if multi_turn_sessions:
            print(f"  Multi-turn sessions: {len(multi_turn_sessions)} (independent runs)")
        
        start_time = time.perf_counter()
        total_queries = 0
        
        # ── 1. Run regular sessions ──
        for session_index, session in enumerate(regular_sessions, start=1):
            session_result = self._run_session(
                session["queries"], session["name"], session_index,
                use_versions=session.get("use_versions", False),
                tree_path=session.get("tree_path"),
            )
            total_queries += len(session_result.queries)
        
        # ── 2. Run multi-turn sessions (each independent, under multi-turn/) ──
        if multi_turn_sessions:
            multi_turn_parent = self.experiment_dir / "multi-turn"
            multi_turn_parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'#'*60}")
            print(f"# Multi-Turn Sessions: {len(multi_turn_sessions)} independent runs")
            print(f"# Output: multi-turn/")
            print(f"{'#'*60}")
            
            for mt_index, mt_session in enumerate(multi_turn_sessions, start=1):
                session_result = self._run_multi_turn_session(
                    turn_queries=mt_session["queries"],
                    session_name=mt_session["name"],
                    multi_turn_parent_dir=multi_turn_parent,
                    session_index=mt_index,
                    tree_path=mt_session.get("tree_path"),
                )
                total_queries += len(session_result.queries)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Save experiment log
        self.result_logger.save_experiment_log(
            experiment_name=self.experiment_dir.name,
            total_sessions=len(self.sessions),
            total_execution_time_ms=total_time
        )
        
        # Build and save cost summary
        cost_summary = self._build_cost_summary(total_time)
        cost_summary_path = self.experiment_dir / "Cost_Summary.json"
        with open(cost_summary_path, "w", encoding="utf-8") as f:
            json.dump(cost_summary, f, indent=2, ensure_ascii=False)
        
        # Calculate totals for summary display
        total_tokens = sum(entry.get("token_usage", {}).get("total_tokens", 0) 
                          for entry in self._cost_tracker)
        total_cost = sum(entry.get("cost_usd", 0) for entry in self._cost_tracker)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Experiment Complete!")
        print(f"{'='*60}")
        print(f"  Regular sessions: {len(regular_sessions)}")
        print(f"  Multi-turn sessions: {len(multi_turn_sessions)}")
        print(f"  Total queries: {total_queries}")
        print(f"  Total time: {total_time/1000:.1f}s")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"\nResults saved to: {self.experiment_dir}")
        print(f"  - Experiment_Log.json")
        print(f"  - Cost_Summary.json")
        
        # Return experiment log
        with open(self.experiment_dir / "Experiment_Log.json", "r") as f:
            return json.load(f)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run in-context evaluation experiment"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="experiment.yaml",
        help="Path to experiment config file (default: experiment.yaml)"
    )
    
    args = parser.parse_args()
    
    runner = InContextRunner(args.config)
    experiment_log = runner.run()
    
    # Print results summary
    print("\n" + "="*60)
    print("Results Summary:")
    print("="*60)
    
    results = experiment_log.get("results", {})
    for query_id, data in list(results.items())[:5]:
        print(f"\n{query_id}:")
        print(f"  Query: {data.get('query', '')[:50]}...")
        print(f"  Operation: {data.get('operation', 'UNKNOWN')}")
        print(f"  Success: {data.get('success', False)}")
        nodes = data.get("top_k_nodes", [])
        if nodes:
            print(f"  Nodes:")
            for node in nodes[:3]:
                print(f"    - {node}")


if __name__ == "__main__":
    main()
