"""
Dense XPath LLM - Executes XPath-like queries with LLM-based semantic matching.

Parses queries like: /Itinerary/Day[2]/Restaurant/[description =~ "cheap"]
- Type matching: /Itinerary, /Day, /Restaurant, /POI
- Index matching: Day[2]
- Semantic predicates: [description =~ "query"]

Supports nested predicates with subtree evidence:
/Itinerary/Day/POI/[description =~ "italian"]/Restaurant/[description =~ "cheap"]
- Earlier predicates can be informed by matches from later predicates in the path
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client import OpenAIClient, get_client
from pipeline_execution.semantic_xpath_execution import LLMClassifier
from pipeline_execution.semantic_xpath_execution import SubtreeRAM


@dataclass
class PathSegment:
    """Represents a single segment in the XPath-like query."""
    node_type: Optional[str] = None  # e.g., "Itinerary", "Day", "POI", "Restaurant"
    index: Optional[int] = None       # e.g., 2 for Day[2]
    predicate: Optional[str] = None   # e.g., "cheap" from [description =~ "cheap"]


@dataclass
class TraceEntry:
    """A single entry in the reasoning trace."""
    step: int
    action: str
    node_type: str
    node_name: Optional[str]
    path_segment: str
    result: str
    details: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    """A matching node with its position in the tree."""
    node: dict
    tree_path: str  # e.g., "/Itinerary/Day[1]/Restaurant[2]"
    
    @property
    def type(self) -> str:
        return self.node.get("type", "unknown")
    
    @property
    def name(self) -> Optional[str]:
        return self.node.get("name")
    
    @property
    def description(self) -> Optional[str]:
        return self.node.get("description")


class DenseXPathLLM:
    """
    Executes XPath-like queries against a tree with LLM-based semantic matching.
    Supports nested predicates with subtree evidence propagation.
    """
    
    TREE_PATH = Path(__file__).parent.parent.parent / "store" / "memory" / "tree_memory.json"
    TRACE_DIR = Path(__file__).parent.parent.parent / "result" / "reasoning_traces"
    
    def __init__(self, client: OpenAIClient = None):
        """
        Initialize the executor.
        
        Args:
            client: Optional OpenAI client. If not provided, one will be created.
        """
        self._client = client
        self._classifier = None  # Lazy load
        self._tree = None  # Lazy load
        self.traces: list[TraceEntry] = []
        self.step_counter = 0
        self.ram: SubtreeRAM = None  # Initialized per query execution
    
    @property
    def client(self) -> OpenAIClient:
        """Lazy load the OpenAI client"""
        if self._client is None:
            self._client = get_client()
        return self._client
    
    @property
    def classifier(self) -> LLMClassifier:
        """Lazy load the LLM classifier"""
        if self._classifier is None:
            self._classifier = LLMClassifier(self.client)
        return self._classifier
    
    @property
    def tree(self) -> dict:
        """Lazy load the tree from file"""
        if self._tree is None:
            with open(self.TREE_PATH, "r") as f:
                self._tree = json.load(f)
        return self._tree
    
    def _add_trace(self, action: str, node: dict, segment_str: str, result: str, **details):
        """Add a trace entry."""
        self.step_counter += 1
        entry = TraceEntry(
            step=self.step_counter,
            action=action,
            node_type=node.get("type", "unknown"),
            node_name=node.get("name"),
            path_segment=segment_str,
            result=result,
            details=details
        )
        self.traces.append(entry)
    
    def _query_to_filename(self, query: str) -> str:
        """Convert a query to a safe filename."""
        safe = re.sub(r'[/\[\]=~"\s]+', '_', query)
        safe = safe.strip('_')
        if len(safe) > 100:
            safe = safe[:100]
        return safe
    
    def _save_traces(self, query: str) -> Path:
        """Save traces to file."""
        self.TRACE_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_name = self._query_to_filename(query)
        filename = f"{query_name}.json"
        filepath = self.TRACE_DIR / filename
        
        trace_data = {
            "query": query,
            "timestamp": timestamp,
            "total_steps": self.step_counter,
            "traces": [
                {
                    "step": t.step,
                    "action": t.action,
                    "node_type": t.node_type,
                    "node_name": t.node_name,
                    "path_segment": t.path_segment,
                    "result": t.result,
                    "details": t.details
                }
                for t in self.traces
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(trace_data, f, indent=2)
        
        return filepath
    
    def parse_query(self, query: str) -> list[PathSegment]:
        """
        Parse an XPath-like query into segments.
        
        Example: /Itinerary/Day[2]/Restaurant[description =~ "cheap"]
        
        Returns list of PathSegment objects.
        """
        segments = []
        
        # Pattern matches:
        # - /Type
        # - /Type[index]
        # - /Type[description =~ "query"]
        # - /Type[index][description =~ "query"]  (index handled separately)
        pattern = r'/([A-Za-z]+)(?:\[(\d+)\])?(?:\[description\s*=~\s*"([^"]+)"\])?'
        
        for match in re.finditer(pattern, query):
            node_type = match.group(1)
            index = int(match.group(2)) if match.group(2) else None
            predicate = match.group(3)  # May be None
            
            # Create type segment (possibly with index)
            segments.append(PathSegment(node_type=node_type, index=index))
            
            # If there's a predicate, add it as a separate segment
            if predicate:
                segments.append(PathSegment(predicate=predicate))
        
        return segments
    
    def _has_more_path_after(self, segments: list[PathSegment], predicate_idx: int) -> bool:
        """Check if there are more path segments after a predicate (making it a deferred predicate)."""
        # Look for any type segment after the predicate
        for i in range(predicate_idx + 1, len(segments)):
            if segments[i].node_type:
                return True
        return False
    
    def _matches_type_segment(self, node: dict, segment: PathSegment, sibling_index: int = None) -> bool:
        """Check if a node matches a type/index segment."""
        if node.get("type") != segment.node_type:
            return False
        
        if segment.index is not None:
            node_index = node.get("index")
            if node_index is not None:
                return node_index == segment.index
            if sibling_index is not None:
                return sibling_index == segment.index
            return False
        
        return True
    
    def _get_all_children(self, node: dict) -> list[dict]:
        """Get all children of a node (flattened, not recursive)."""
        return node.get("children", [])
    
    def _matches_predicate(
        self, 
        node: dict, 
        predicate: str, 
        has_subtree_evidence: bool = False
    ) -> bool:
        """
        Check if a node matches a semantic predicate.
        
        Args:
            node: The node to check
            predicate: The semantic query
            has_subtree_evidence: Whether this node passed pruning (has matching descendants)
        """
        # Always pass full subtree for evaluation (all children, not just matched ones)
        full_subtree = self._get_all_children(node)
        
        if full_subtree:
            # Parent node: evaluate with full subtree context
            return self.classifier.classify_node(node, predicate, full_subtree)
        else:
            # Leaf node: only node's own description
            return self.classifier.classify_node(node, predicate)
    
    def _build_tree_path(self, node: dict, sibling_index: int = None, parent_path: str = "") -> str:
        """Build the actual tree path with indices for the node."""
        node_type = node.get("type", "unknown")
        index = node.get("index", sibling_index)
        if index is not None:
            return f"{parent_path}/{node_type}[{index}]"
        return f"{parent_path}/{node_type}"
    
    def _segment_to_string(self, segment: PathSegment) -> str:
        """Convert a segment to string representation."""
        if segment.node_type:
            if segment.index:
                return f"{segment.node_type}[{segment.index}]"
            return segment.node_type
        if segment.predicate:
            return f'[description =~ "{segment.predicate}"]'
        return "?"
    
    def _execute_dfs(
        self,
        node: dict,
        segments: list[PathSegment],
        segment_idx: int,
        sibling_index: int = None,
        path: str = "",
        tree_path: str = ""
    ) -> list[MatchResult]:
        """
        DFS traversal to find matching nodes.
        Supports deferred predicate evaluation with subtree evidence.
        
        Args:
            node: Current node
            segments: List of path segments
            segment_idx: Current segment index
            sibling_index: 1-based index among same-type siblings
            path: Current path string for tracing (query path)
            tree_path: Actual position in tree
            
        Returns:
            List of MatchResult objects with node and tree path
        """
        if segment_idx >= len(segments):
            return []
        
        current_segment = segments[segment_idx]
        segment_str = self._segment_to_string(current_segment)
        current_path = f"{path}/{segment_str}"
        current_tree_path = self._build_tree_path(node, sibling_index, tree_path)
        
        # === CASE 1: Type/Index segment ===
        if current_segment.node_type:
            matches = self._matches_type_segment(node, current_segment, sibling_index)
            
            action = "index_match" if current_segment.index else "type_match"
            self._add_trace(
                action=action,
                node=node,
                segment_str=segment_str,
                result="MATCH" if matches else "NO_MATCH",
                current_path=current_path,
                tree_path=current_tree_path,
                segment_index=segment_idx,
                sibling_index=sibling_index
            )
            
            if not matches:
                return []
            
            next_idx = segment_idx + 1
            
            # Check if next segment is a predicate
            if next_idx < len(segments) and segments[next_idx].predicate:
                predicate_segment = segments[next_idx]
                predicate_str = self._segment_to_string(predicate_segment)
                
                # Check if this is a DEFERRED predicate (more path after it)
                is_deferred = self._has_more_path_after(segments, next_idx)
                
                if is_deferred:
                    # === DEFERRED PREDICATE: First collect subtree matches ===
                    self._add_trace(
                        action="deferred_predicate_start",
                        node=node,
                        segment_str=predicate_str,
                        result="DEFERRED",
                        current_path=f"{current_path}/{predicate_str}",
                        tree_path=current_tree_path,
                        segment_index=next_idx,
                        reason="More path segments after predicate - collecting subtree evidence first"
                    )
                    
                    # Continue traversal to get subtree matches (skip the predicate for now)
                    subtree_results = []
                    children = node.get("children", [])
                    type_counts = {}
                    
                    for child in children:
                        child_type = child.get("type")
                        if child_type not in type_counts:
                            type_counts[child_type] = 0
                        type_counts[child_type] += 1
                        child_sibling_idx = type_counts[child_type]
                        
                        child_results = self._execute_dfs(
                            child,
                            segments,
                            next_idx + 1,  # Skip the predicate, continue with path
                            child_sibling_idx,
                            f"{current_path}/{predicate_str}",
                            current_tree_path
                        )
                        subtree_results.extend(child_results)
                    
                    # Store subtree in RAM for this parent node
                    if subtree_results:
                        self.ram.store(
                            tree_path=current_tree_path,
                            node=node,
                            predicate=predicate_segment.predicate,
                            matched_children=[r.node for r in subtree_results]
                        )
                    
                    # Now evaluate the deferred predicate with FULL subtree
                    # (pruning already done by checking subtree_results exists)
                    if subtree_results:
                        predicate_matches = self._matches_predicate(
                            node, 
                            predicate_segment.predicate,
                            has_subtree_evidence=True  # Pass full subtree internally
                        )
                        
                        self._add_trace(
                            action="deferred_predicate_eval",
                            node=node,
                            segment_str=predicate_str,
                            result="MATCH" if predicate_matches else "NO_MATCH",
                            current_path=f"{current_path}/{predicate_str}",
                            tree_path=current_tree_path,
                            segment_index=next_idx,
                            pruning_evidence_count=len(subtree_results),
                            pruning_evidence=[r.tree_path for r in subtree_results[:5]],  # First 5
                            full_subtree_count=len(node.get("children", [])),
                            ram_stored=True
                        )
                        
                        if predicate_matches:
                            # Return the subtree results (they are the final matches)
                            return subtree_results
                    else:
                        self._add_trace(
                            action="deferred_predicate_eval",
                            node=node,
                            segment_str=predicate_str,
                            result="NO_MATCH",
                            current_path=f"{current_path}/{predicate_str}",
                            tree_path=current_tree_path,
                            segment_index=next_idx,
                            reason="No subtree matches found"
                        )
                    
                    return []
                
                else:
                    # === TERMINAL PREDICATE: Evaluate locally ===
                    predicate_matches = self._matches_predicate(node, predicate_segment.predicate)
                    
                    self._add_trace(
                        action="semantic_match",
                        node=node,
                        segment_str=predicate_str,
                        result="MATCH" if predicate_matches else "NO_MATCH",
                        current_path=f"{current_path}/{predicate_str}",
                        tree_path=current_tree_path,
                        segment_index=next_idx
                    )
                    
                    if not predicate_matches:
                        return []
                    
                    next_idx += 1
            
            # If we've processed all segments, return this node
            if next_idx >= len(segments):
                self._add_trace(
                    action="result_found",
                    node=node,
                    segment_str=segment_str,
                    result="COLLECTED",
                    tree_path=current_tree_path,
                    node_details={
                        "type": node.get("type"),
                        "name": node.get("name"),
                        "description": node.get("description", "")[:100]
                    }
                )
                return [MatchResult(node=node, tree_path=current_tree_path)]
            
            # Continue to children
            results = []
            children = node.get("children", [])
            type_counts = {}
            
            for child in children:
                child_type = child.get("type")
                if child_type not in type_counts:
                    type_counts[child_type] = 0
                type_counts[child_type] += 1
                child_sibling_idx = type_counts[child_type]
                
                child_results = self._execute_dfs(
                    child,
                    segments,
                    next_idx,
                    child_sibling_idx,
                    current_path,
                    current_tree_path
                )
                results.extend(child_results)
            
            return results
        
        # === CASE 2: Standalone predicate (shouldn't happen in well-formed queries) ===
        return []
    
    def execute(self, query: str, save_trace: bool = True, save_ram: bool = True) -> list[MatchResult]:
        """
        Execute an XPath-like query against the tree.
        
        Args:
            query: XPath-like query string
            save_trace: Whether to save reasoning traces
            save_ram: Whether to save RAM state to disk
            
        Returns:
            List of MatchResult objects with node and tree path
        """
        # Reset state
        self.traces = []
        self.step_counter = 0
        self.ram = SubtreeRAM(query_id=self._query_to_filename(query))
        
        # Parse query
        segments = self.parse_query(query)
        
        if not segments:
            return []
        
        self._add_trace(
            action="parse_query",
            node={"type": "query"},
            segment_str=query,
            result="PARSED",
            segments=[self._segment_to_string(s) for s in segments]
        )
        
        # Start DFS from root
        results = self._execute_dfs(self.tree, segments, 0, sibling_index=1)
        
        # Save traces
        if save_trace:
            trace_file = self._save_traces(query)
            print(f"Trace saved to: {trace_file}")
        
        # Save RAM state
        if save_ram and len(self.ram) > 0:
            ram_file = self.ram.save_to_disk()
            print(f"RAM saved to: {ram_file}")
        
        return results


# Convenience function
def execute_dense_xpath(query: str, client: OpenAIClient = None) -> list[MatchResult]:
    """
    Execute an XPath-like query with LLM semantic matching.
    
    Args:
        query: XPath-like query string
        client: Optional OpenAI client
        
    Returns:
        List of MatchResult objects with node and tree path
    """
    executor = DenseXPathLLM(client)
    return executor.execute(query)


if __name__ == "__main__":
    executor = DenseXPathLLM()
    
    test_queries = [
        # Simple query (terminal predicate)
        "/Itinerary/Day[1]/Restaurant/[description =~ \"jazz\"]",
        # Nested query (deferred predicate with subtree evidence)
        "/Itinerary/Day/POI/[description =~ \"italian\"]/Restaurant/[description =~ \"upscale\"]",
    ]
    
    print("Testing Dense XPath LLM")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        results = executor.execute(query)
        print(f"Found {len(results)} matching nodes:")
        for result in results:
            print(f"  - [{result.tree_path}]")
            print(f"    {result.type}: {result.name}")
        print()
