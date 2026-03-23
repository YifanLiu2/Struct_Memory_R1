"""
XPath Pipeline - Main orchestration for XPath query generation and execution.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from client import get_client
from xpath_query_generation import XPathQueryGenerator
from pipeline_execution.semantic_xpath_execution import DenseXPathExecutor, ExecutionTrace
from predicate_classifiers import NodeInfo, LLMPredicateClassifier, EntailmentPredicateClassifier, CosinePredicateClassifier


class XPathPipeline:
    """
    Pipeline for converting user requests into XPath-like queries and executing them.
    Logs all interactions to reasoning traces.
    
    Supports three classifier types:
    - "llm": LLM-based classification (default)
    - "entailment": BART-NLI entailment scoring
    - "cosine": TAS-B cosine similarity
    """
    
    LOG_DIR = Path(__file__).parent.parent / "reasoning_traces" / "logs"
    
    def __init__(self, client=None, classifier_type: str = "llm"):
        """
        Initialize the pipeline.
        
        Args:
            client: Optional OpenAI client. If not provided, one will be created.
            classifier_type: "llm" or "entailment". Defaults to "llm".
        """
        self.client = client or get_client()
        self.classifier_type = classifier_type
        
        # Components with shared client
        self.query_generator = XPathQueryGenerator(client=self.client)
        
        # Create classifier based on type
        classifier = self._create_classifier(classifier_type)
        self.executor = DenseXPathExecutor(classifier=classifier)
    
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.query_history = []
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info(f"Pipeline initialized. Session ID: {self.session_id}")
        self.logger.info(f"Classifier type: {self.classifier_type}")
    
    def _create_classifier(self, classifier_type: str):
        """Create the appropriate classifier based on type."""
        if classifier_type == "llm":
            return LLMPredicateClassifier(client=self.client)
        elif classifier_type == "entailment":
            return EntailmentPredicateClassifier()
        elif classifier_type == "cosine":
            return CosinePredicateClassifier()
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}. Use 'llm', 'entailment', or 'cosine'.")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup file and console logging."""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(f"xpath_pipeline_{self.session_id}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        
        # File handler
        log_file = self.LOG_DIR / f"pipeline_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        self.log_file = log_file
        return logger
    
    def generate_query(self, user_request: str) -> str:
        """
        Convert a user request into an XPath-like query.
        
        Args:
            user_request: Natural language request from the user
            
        Returns:
            XPath-like query string
        """
        self.logger.info(f"User request: {user_request}")
        
        xpath_query = self.query_generator.generate(user_request)
        
        self.logger.info(f"Generated XPath: {xpath_query}")
        
        return xpath_query
    
    def execute_query(self, xpath_query: str) -> tuple[list[NodeInfo], ExecutionTrace]:
        """
        Execute an XPath-like query.
        
        Args:
            xpath_query: XPath-like query string
            
        Returns:
            Tuple of (matching nodes, execution trace)
        """
        self.logger.info(f"Executing query: {xpath_query}")
        
        results, trace = self.executor.execute(xpath_query, save_trace=True)
        
        self.logger.info(f"Found {len(results)} matching nodes")
        for r in results:
            self.logger.info(f"  - {r.tree_path}: {r.name}")
        
        return results, trace
    
    def run(self, user_request: str) -> tuple[list[NodeInfo], str, ExecutionTrace]:
        """
        Run the full pipeline: generate query and execute it.
        
        Args:
            user_request: Natural language request
            
        Returns:
            Tuple of (matching nodes, xpath query, execution trace)
        """
        self.logger.info("=" * 50)
        self.logger.info("Pipeline run started")
        
        # Step 1: Generate XPath query
        xpath_query = self.generate_query(user_request)
        
        # Step 2: Execute query
        results, trace = self.execute_query(xpath_query)
        
        # Record in history
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_request": user_request,
            "xpath_query": xpath_query,
            "result_count": len(results),
            "results": [
                {"path": r.tree_path, "name": r.name, "type": r.node_type}
                for r in results
            ]
        }
        self.query_history.append(entry)
        
        # Log trace summary
        self._log_trace(entry)
        
        self.logger.info("Pipeline run completed")
        self.logger.info("=" * 50)
        
        return results, xpath_query, trace
    
    def _log_trace(self, entry: dict):
        """Log a query trace."""
        trace_line = json.dumps(entry, ensure_ascii=False)
        self.logger.debug(f"TRACE: {trace_line}")
    
    def save_session(self) -> Path:
        """Save the complete session history."""
        session_file = self.LOG_DIR / f"session_{self.session_id}.json"
        
        session_data = {
            "session_id": self.session_id,
            "start_time": self.query_history[0]["timestamp"] if self.query_history else None,
            "end_time": datetime.now().isoformat(),
            "total_queries": len(self.query_history),
            "queries": self.query_history
        }
        
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Session saved to: {session_file}")
        return session_file
    
    def run_cli(self):
        """Run interactive CLI mode."""
        print("\n" + "=" * 60)
        print("  XPath Pipeline - Interactive Mode")
        print("=" * 60)
        print(f"  Session ID: {self.session_id}")
        print(f"  Classifier: {self.classifier_type}")
        print(f"  Log file: {self.log_file}")
        print("  Commands:")
        print("    quit/exit - Exit and save session")
        print("    history   - Show query history")
        print("    save      - Save session")
        print("    query     - Generate query only (no execution)")
        print("=" * 60 + "\n")
        
        self.logger.info("CLI session started")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Commands
                if user_input.lower() in ("quit", "exit", "q"):
                    self.save_session()
                    print("\nSession saved. Goodbye!")
                    self.logger.info("CLI session ended")
                    break
                
                if user_input.lower() == "history":
                    self._print_history()
                    continue
                
                if user_input.lower() == "save":
                    session_file = self.save_session()
                    print(f"Session saved to: {session_file}")
                    continue
                
                if user_input.lower().startswith("query "):
                    # Query generation only
                    request = user_input[6:].strip()
                    xpath_query = self.generate_query(request)
                    print(f"XPath: {xpath_query}\n")
                    continue
                
                # Full pipeline
                print("-" * 40)
                results, xpath_query, trace = self.run(user_input)
                
                print(f"XPath: {xpath_query}")
                print(f"\n✓ Found {len(results)} matching node(s):")
                print("-" * 40)
                
                for i, r in enumerate(results, 1):
                    print(f"  [{i}] {r.tree_path}")
                    print(f"      {r.node_type}: {r.name}")
                    if r.description:
                        desc = r.description[:80] + "..." if len(r.description) > 80 else r.description
                        print(f"      └─ {desc}")
                
                if not results:
                    print("  (no matches found)")
                
                print("-" * 40 + "\n")
                
            except KeyboardInterrupt:
                self.save_session()
                print("\n\nSession saved. Goodbye!")
                self.logger.info("CLI session interrupted")
                break
            except Exception as e:
                error_msg = f"Error: {e}"
                print(error_msg)
                self.logger.error(error_msg, exc_info=True)
    
    def _print_history(self):
        """Print query history."""
        print("\n" + "-" * 40)
        print("Query History:")
        print("-" * 40)
        
        if not self.query_history:
            print("  (no queries yet)")
        else:
            for i, entry in enumerate(self.query_history, 1):
                print(f"  [{i}] {entry['user_request']}")
                print(f"      → {entry['xpath_query']}")
                print(f"      → {entry['result_count']} result(s)")
        
        print("-" * 40 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XPath Pipeline - Semantic Query Engine")
    parser.add_argument(
        "--classifier", "-c",
        type=str,
        choices=["llm", "entailment", "cosine"],
        default="llm",
        help="Classifier type: 'llm' (default), 'entailment', or 'cosine'"
    )
    args = parser.parse_args()
    
    pipeline = XPathPipeline(classifier_type=args.classifier)
    pipeline.run_cli()

