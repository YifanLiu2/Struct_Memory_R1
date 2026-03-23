"""
Semantic XPath Pipeline - Core CRUD Pipeline for tree operations.

Provides the core pipeline logic for executing CRUD operations on tree data
with full query execution, trace saving, and version management.

Uses in-tree versioning - all operations create new Version nodes within the tree,
enabling full history tracking and version-based queries.

Supports:
- Read: Find and retrieve nodes using semantic XPath
- Create: Add new nodes to the tree (creates new version)
- Update: Modify existing nodes (creates new version)
- Delete: Remove nodes from the tree (creates new version)
"""

import time
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline_execution.pipeline_orchestrator.pipeline_orchestrator import SemanticXPathOrchestrator
from pipeline_execution.semantic_xpath_execution import TraceWriter
from pipeline.semantic_xpath_pipeline.semantic_xpath_data_model import SessionStatistics


class SemanticXPathPipeline:
    """
    Full CRUD Pipeline for semantic XPath operations with in-tree versioning.
    
    Converts natural language requests to CRUD operations:
    - Classifies intent and generates XPath in a single LLM call
    - Executes operations with LLM reasoning
    - Creates new versions within the tree for modifications
    - Provides full query display (e.g., "Delete(/Itinerary/Version[-1]/Day/POI[...])")
    
    All modifications create new Version nodes in the tree, enabling:
    - Full history tracking
    - Version-based queries ("show me the second version")
    - Semantic version search ("what did I change about the museum?")
    """
    
    def __init__(
        self, 
        top_k: int = None, 
        score_threshold: float = None,
        scoring_method: str = None,
        tree_path: Path = None,
        traces_path: Path = None,
        config: dict = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            top_k: Number of top-scoring nodes to consider for semantic predicates.
                   If None, uses value from config.
            score_threshold: Minimum score for a node to be considered relevant.
                   If None, uses value from config.
            scoring_method: Scoring method ("llm" or "entailment").
                   If None, uses value from config.
            tree_path: Optional path to the XML tree. Overrides config default.
            traces_path: Optional path for trace files. If None, uses default traces folder.
            config: Optional config dict. If not provided, loads from config.yaml.
        """
        self._traces_path = traces_path
        self.orchestrator = SemanticXPathOrchestrator(
            scoring_method=scoring_method,
            top_k=top_k,
            score_threshold=score_threshold,
            tree_path=tree_path,
            traces_path=traces_path,
            config=config
        )
        self.trace_writer = TraceWriter(
            traces_path=traces_path / "reasoning_traces" if traces_path else None
        )
        
        # Session statistics
        self.session_stats = SessionStatistics()
    
    def set_traces_path(self, traces_path: Path):
        """
        Update the traces path for all components.
        
        Used to redirect traces to per-query folders during experiments.
        
        Args:
            traces_path: New directory for trace files
        """
        self._traces_path = traces_path
        reasoning_traces_path = traces_path / "reasoning_traces" if traces_path else None
        
        # Update trace writer
        self.trace_writer = TraceWriter(traces_path=reasoning_traces_path)

        self.orchestrator.executor.trace_writer = TraceWriter(traces_path=reasoning_traces_path)
        
        # Update scorer's traces path
        if hasattr(self.orchestrator.executor.scorer, 'traces_path'):
            self.orchestrator.executor.scorer.traces_path = reasoning_traces_path
            if reasoning_traces_path:
                reasoning_traces_path.mkdir(parents=True, exist_ok=True)
        
        # Update handlers' traces paths
        for handler in [self.orchestrator.read_handler, self.orchestrator.delete_handler,
                       self.orchestrator.update_handler, self.orchestrator.create_handler]:
            if hasattr(handler, 'traces_path'):
                handler.traces_path = reasoning_traces_path
                if reasoning_traces_path:
                    reasoning_traces_path.mkdir(parents=True, exist_ok=True)
    
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process a user request as a CRUD operation.
        
        Args:
            user_request: Natural language request from the user
            
        Returns:
            Dict with operation results, traces, and timing info
        """
        start_time = time.perf_counter()

        result = self.orchestrator.execute(user_request)
        
        # Calculate total pipeline timing
        total_time_ms = (time.perf_counter() - start_time) * 1000
        result["total_time_ms"] = total_time_ms
        
        # Preserve step timing from executor, add total
        if "timing" not in result:
            result["timing"] = {}
        result["timing"]["pipeline_total_ms"] = total_time_ms
        
        # Update session stats
        self.session_stats.update(result)
        
        # Save CRUD operation traces
        if "timestamp" in result:
            self.trace_writer.save_crud_traces(result["timestamp"], result)
        
        return result
