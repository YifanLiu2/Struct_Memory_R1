"""
Semantic XPath CLI - Interactive command-line interface.

Provides an interactive shell for executing CRUD operations with:
- Natural language query processing
- Session statistics tracking
- Version history viewing
- Real-time result display
- Session-based logging with per-query trace folders
"""

import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from pipeline.semantic_xpath_pipeline.semantic_xpath_pipeline import SemanticXPathPipeline
from pipeline.semantic_xpath_pipeline.semantic_xpath_data_model import ResultFormatter
from utils.logger.session_logging.session_manager import SessionManager


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class SemanticXPathCLI:
    """
    Interactive CLI for Semantic XPath operations.
    
    Provides a command-line interface with:
    - Natural language CRUD operations
    - Session statistics and summaries
    - Version history management
    - Tree reload functionality
    - Session-based logging with per-query trace folders
    
    When session logging is enabled, the CLI:
    1. Copies the original tree to session folder at start
    2. Creates a new pipeline that operates on this session tree
    3. All modifications accumulate in the session tree (evolving tree)
    4. No per-query tree snapshots - just one evolving tree per session
    """
    
    def __init__(
        self, 
        pipeline_config: Optional[Dict[str, Any]] = None,
        session_manager: Optional[SessionManager] = None,
        enable_session_logging: bool = True
    ):
        """
        Initialize the CLI.
        
        Args:
            pipeline_config: Configuration for creating the pipeline (top_k, score_threshold, scoring_method)
            session_manager: Optional SessionManager for logging (created if not provided)
            enable_session_logging: Whether to enable session-based logging (default: True)
        """
        self.pipeline_config = pipeline_config or {}
        self.pipeline: Optional[SemanticXPathPipeline] = None
        self.formatter = ResultFormatter()
        self.enable_session_logging = enable_session_logging
        
        # Initialize session manager for logging
        if enable_session_logging:
            self.session_manager = session_manager or SessionManager()
        else:
            self.session_manager = None
        
        self._session_tree_path: Optional[Path] = None
        self._original_tree_path: Optional[Path] = None
    
    def run_interactive(self):
        """
        Run an interactive loop for CRUD operations.
        
        Commands:
        - Type a natural language query to execute a CRUD operation
        - Type 'stats' to see session statistics
        - Type 'history' to see version history
        - Type 'reload' to reload the tree from the original file
        - Type 'exit' or 'quit' to stop
        
        Session logging flow:
        1. Copy original tree to session folder
        2. Create pipeline that operates on session tree (evolving tree)
        3. All modifications accumulate in session tree
        4. Query traces saved to per-query folders
        """
        self._print_welcome()
        session_start = time.perf_counter()
        
        # Determine tree path for the pipeline
        tree_path = None
        
        if self.session_manager:
            # Start session and set up session tree
            session_dir = self.session_manager.start_session()
            print(f"📁 Session logs: {session_dir}")
            
            # Get original tree path from a temporary pipeline to find the default tree
            temp_pipeline = SemanticXPathPipeline(**self.pipeline_config)
            self._original_tree_path = temp_pipeline.orchestrator.tree_path
            
            # Copy the source tree to session folder
            self._session_tree_path = self.session_manager.copy_tree_to_session(self._original_tree_path)
            print(f"📄 Session tree: {self._session_tree_path}")
            print()
            
            # Use session tree for the main pipeline
            tree_path = self._session_tree_path
        
        # Create the pipeline (with session tree if logging enabled)
        self.pipeline = SemanticXPathPipeline(
            tree_path=tree_path,
            **self.pipeline_config
        )
        
        while True:
            try:
                user_input = input("🔄 Query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("exit", "quit", "q"):
                    self._end_session(session_start)
                    print("Goodbye!")
                    break
                
                if user_input.lower() == "stats":
                    self._print_stats()
                    continue
                
                if user_input.lower() == "history":
                    self._print_version_history()
                    continue
                
                if user_input.lower() == "reload":
                    self.pipeline.orchestrator.reload_tree()
                    print("✅ Tree reloaded from original file")
                    continue
                
                # Start query logging if enabled
                query_dir = None
                if self.session_manager:
                    query_dir = self.session_manager.start_query()
                    # Set traces path to query directory for this query
                    self.pipeline.set_traces_path(query_dir)
                
                # Process the query (modifications go directly to session tree)
                result = self.pipeline.process_request(user_input)
                print(self.formatter.format_result(result))
                
                # Save query logs if enabled (no per-query tree snapshots)
                if self.session_manager and query_dir:
                    self.session_manager.save_query_log(user_input, result, query_dir)
                    print(f"📝 Query logs saved: {query_dir}")
                
                print()
                
            except KeyboardInterrupt:
                self._end_session(session_start)
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    def _end_session(self, session_start: float):
        """End the session and save summary.
        
        The session tree already contains all modifications (evolving tree),
        no need to save separately - pipeline's version_manager handles saves.
        """
        self._print_session_summary(session_start)
        
        # End session logging if enabled
        if self.session_manager and self.pipeline:
            stats = self.pipeline.session_stats.to_dict()
            self.session_manager.end_session(stats)
    
    def run_single_query(self, query: str):
        """
        Execute a single query and print the result.
        
        Args:
            query: Natural language query to execute
        """
        # Create pipeline if not already created (for single query mode)
        if self.pipeline is None:
            self.pipeline = SemanticXPathPipeline(**self.pipeline_config)
        
        result = self.pipeline.process_request(query)
        print(self.formatter.format_result(result))
    
    def _print_welcome(self):
        """Print welcome message and instructions."""
        print("=" * 60)
        print("Semantic XPath Pipeline - CRUD Operations")
        print("=" * 60)
        print("In-tree versioning enabled:")
        print("  - All modifications create new versions")
        print("  - Query specific versions with Version[N] or Version[-1]")
        print("  - Search versions: 'what changed about museum?'")
        print("-" * 60)
        print("Commands:")
        print("  - Natural language query for CRUD operations")
        print("  - 'stats' - Session statistics")
        print("  - 'history' - View version history")
        print("  - 'reload' - Reload tree from file")
        print("  - 'exit' or 'quit' - Exit")
        print("=" * 60)
        print()
    
    def _print_stats(self):
        """Print session statistics."""
        stats = self.pipeline.session_stats.to_dict()
        
        print("\n📊 Session Statistics:")
        print("-" * 40)
        print(f"  Total Operations: {stats['operations']}")
        print(f"  - Reads:   {stats['reads']}")
        print(f"  - Creates: {stats['creates']}")
        print(f"  - Updates: {stats['updates']}")
        print(f"  - Deletes: {stats['deletes']}")
        print(f"  Successes: {stats['successes']}")
        print(f"  Failures:  {stats['failures']}")
        print(f"  Versions Created: {stats['versions_created']}")
        print()
    
    def _print_version_history(self):
        """Print version history from the tree."""
        print("\n📜 Version History:")
        print("-" * 50)
        
        history = self.pipeline.orchestrator.version_manager.get_version_history(
            self.pipeline.orchestrator.tree
        )
        
        if not history:
            print("  No versions found")
            return
        
        for version in history:
            print(f"\n  Version {version['number']}:")
            if version['patch_info']:
                print(f"    📝 Changes: {version['patch_info']}")
            if version['conversation_history']:
                print(f"    💬 Request: {version['conversation_history']}")
            print(f"    📦 Content: {version['content_count']} items")
        print()
    
    def _print_session_summary(self, session_start: float):
        """
        Print session summary on exit.
        
        Args:
            session_start: Session start time (from time.perf_counter())
        """
        session_time = (time.perf_counter() - session_start) * 1000
        stats = self.pipeline.session_stats.to_dict()
        
        print("\n" + "=" * 60)
        print("📊 Session Summary:")
        print("-" * 40)
        print(f"  Duration: {session_time/1000:.1f}s")
        print(f"  Operations: {stats['operations']}")
        print(f"  - Reads:   {stats['reads']}")
        print(f"  - Creates: {stats['creates']}")
        print(f"  - Updates: {stats['updates']}")
        print(f"  - Deletes: {stats['deletes']}")
        print(f"  Versions Created: {stats['versions_created']}")
        
        if stats['operations'] > 0:
            print(f"  Success Rate: {stats['success_rate']:.1f}%")
        
        print("=" * 60)


def main():
    """Main entry point for running the pipeline interactively."""
    config = load_config()
    executor_config = config.get("xpath_executor", {})
    default_top_k = executor_config.get("top_k", 5)
    default_threshold = executor_config.get("score_threshold", 0.5)
    default_method = executor_config.get("scoring_method", "entailment")
    
    parser = argparse.ArgumentParser(description="Semantic XPath Pipeline - CRUD Operations")
    parser.add_argument("--top-k", type=int, default=None, 
                        help=f"Top K nodes for semantic matching (default from config: {default_top_k})")
    parser.add_argument("--threshold", type=float, default=None, 
                        help=f"Score threshold for relevance (default from config: {default_threshold})")
    parser.add_argument("--scoring", "-s", type=str, default=None,
                        choices=["llm", "entailment", "cosine"],
                        help=f"Scoring method: llm, entailment, or cosine (default from config: {default_method})")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Single query to execute (non-interactive)")
    parser.add_argument("--no-session-log", action="store_true",
                        help="Disable session-based logging to cli_session_results/")
    
    args = parser.parse_args()
    
    # Build pipeline config (pipeline created later in CLI)
    pipeline_config = {
        "top_k": args.top_k,
        "score_threshold": args.threshold,
        "scoring_method": args.scoring
    }
    
    cli = SemanticXPathCLI(
        pipeline_config=pipeline_config,
        enable_session_logging=not args.no_session_log
    )
    
    if args.query:
        # Single query mode (without session logging)
        cli.run_single_query(args.query)
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()
