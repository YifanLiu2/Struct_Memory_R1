"""
LLM XPath Pipeline - Main orchestration class for XPath query generation and execution.
"""

import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from client import OpenAIClient, get_client
from query_generation import XPathQueryGenerator
from pipeline_execution.semantic_xpath_execution import DenseXPathLLM, MatchResult


class LLMXPathPipeline:
    """
    Main pipeline for converting user requests into XPath-like queries
    and executing them against a tree structure.
    """
    
    LOG_DIR = Path(__file__).parent.parent / "store" / "log"
    
    def __init__(self, client: OpenAIClient = None):
        """
        Initialize the pipeline.
        
        Args:
            client: Optional OpenAI client. If not provided, one will be created.
        """
        self.client = client if client is not None else get_client()
        
        # Query generator
        self.query_generator = XPathQueryGenerator()
        self.query_generator._client = self.client
        
        # XPath executor
        self.executor = DenseXPathLLM(self.client)
        
        # Setup logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup file and console logging"""
        # Ensure log directory exists
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger("llm_xpath_pipeline")
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # File handler - one log file per session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.LOG_DIR / f"pipeline_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Store log file path for reference
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
        
        query = self.query_generator.generate(user_request)
        
        self.logger.info(f"Generated query: {query}")
        
        return query
    
    def execute_query(self, xpath_query: str, save_trace: bool = True) -> list[MatchResult]:
        """
        Execute an XPath-like query against the tree.
        
        Args:
            xpath_query: XPath-like query string
            save_trace: Whether to save reasoning traces
            
        Returns:
            List of MatchResult objects with node and tree path
        """
        self.logger.info(f"Executing query: {xpath_query}")
        
        results = self.executor.execute(xpath_query, save_trace=save_trace)
        
        self.logger.info(f"Found {len(results)} matching nodes")
        for result in results:
            self.logger.info(f"  - {result.tree_path} | {result.type}: {result.name}")
        
        return results
    
    def run(self, user_request: str, save_trace: bool = True) -> list[MatchResult]:
        """
        Run the full pipeline: convert user request to query and execute it.
        
        Args:
            user_request: Natural language request from the user
            save_trace: Whether to save reasoning traces
            
        Returns:
            List of MatchResult objects with node and tree path
        """
        self.logger.info("=" * 50)
        self.logger.info(f"Pipeline run started")
        
        # Step 1: Generate XPath query from user request
        xpath_query = self.generate_query(user_request)
        
        # Step 2: Execute the query
        results = self.execute_query(xpath_query, save_trace=save_trace)
        
        self.logger.info(f"Pipeline run completed")
        self.logger.info("=" * 50)
        
        return results
    
    def run_cli(self):
        """
        Run interactive CLI mode.
        """
        print("\n" + "=" * 60)
        print("  LLM XPath Pipeline - Interactive Mode")
        print("=" * 60)
        print(f"  Log file: {self.log_file}")
        print("  Type 'quit' or 'exit' to stop")
        print("  Type 'help' for examples")
        print("=" * 60 + "\n")
        
        self.logger.info("CLI session started")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("quit", "exit", "q"):
                    print("\nGoodbye!")
                    self.logger.info("CLI session ended by user")
                    break
                
                if user_input.lower() == "help":
                    self._print_help()
                    continue
                
                # Run full pipeline
                print("\n" + "-" * 40)
                
                # Step 1: Generate query
                xpath_query = self.generate_query(user_input)
                print(f"Generated Query: {xpath_query}")
                
                # Step 2: Execute query
                print("\nExecuting query...")
                results = self.execute_query(xpath_query)
                
                # Step 3: Display results
                print(f"\n✓ Found {len(results)} matching node(s):")
                print("-" * 40)
                for i, result in enumerate(results, 1):
                    print(f"  [{i}] {result.tree_path}")
                    print(f"      {result.type}: {result.name}")
                    if result.description:
                        print(f"      └─ {result.description[:80]}...")
                
                if not results:
                    print("  (no matches found)")
                
                print("-" * 40 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                self.logger.info("CLI session interrupted")
                break
            except Exception as e:
                error_msg = f"Error: {e}"
                print(error_msg)
                self.logger.error(error_msg, exc_info=True)
    
    def _print_help(self):
        """Print help with example requests"""
        print("\n" + "-" * 40)
        print("Example requests:")
        print("-" * 40)
        examples = [
            "italian related places in all days",
            "italian places in first day",
            "cheap restaurants in day 2",
            "cheap restaurants in days that have italian places",
        ]
        for ex in examples:
            print(f"  • {ex}")
        print("-" * 40 + "\n")


# Convenience function
def create_pipeline(client: OpenAIClient = None) -> LLMXPathPipeline:
    """
    Create a new LLM XPath Pipeline instance.
    
    Args:
        client: Optional OpenAI client
        
    Returns:
        LLMXPathPipeline instance
    """
    return LLMXPathPipeline(client)


if __name__ == "__main__":
    pipeline = LLMXPathPipeline()
    pipeline.run_cli()

