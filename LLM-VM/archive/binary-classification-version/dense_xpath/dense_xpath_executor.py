"""
Dense XPath Executor - Executes XPath-like queries against XML tree.

Supports:
- Type matching: /Itinerary, /Day, /POI, /Restaurant
- Index matching: Day[1], Day[2]
- Semantic predicates: [description =~ "query"]

Algorithm:
1. Top-down: traverse XML matching types and indices to prune
2. Bottom-up: for predicates, collect leaf nodes, batch classify
3. For parent predicates: after matching children, batch classify parents with subtree context
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from predicate_classifiers import (
    LLMPredicateClassifier, 
    NodeInfo, 
    ClassificationResult,
    BatchClassificationResult
)


@dataclass
class PathSegment:
    """Represents a single segment in the XPath-like query."""
    node_type: Optional[str] = None  # e.g., "Itinerary", "Day", "POI", "Restaurant"
    index: Optional[int] = None       # e.g., 1 for Day[1]
    predicate: Optional[str] = None   # e.g., "italian" from [description =~ "italian"]


@dataclass
class ExecutionStep:
    """A single step in the execution trace."""
    step_num: int
    action: str  # "type_match", "index_match", "predicate_batch", "result"
    segment: str
    nodes_before: int
    nodes_after: int
    details: dict = field(default_factory=dict)
    duration_ms: Optional[float] = None  # Duration of this step (for predicate_batch steps)


@dataclass
class ExecutionTrace:
    """Complete trace of query execution."""
    query: str
    timestamp: str
    segments: list[str]
    steps: list[ExecutionStep] = field(default_factory=list)
    batch_classifications: list[dict] = field(default_factory=list)  # Detailed batch info
    results: list[dict] = field(default_factory=list)
    total_duration_ms: Optional[float] = None  # Total execution time
    classification_duration_ms: Optional[float] = None  # Total time spent in classification
    
    def to_dict(self) -> dict:
        result = {
            "query": self.query,
            "timestamp": self.timestamp,
            "segments": self.segments,
            "steps": [
                {
                    "step": s.step_num,
                    "action": s.action,
                    "segment": s.segment,
                    "nodes_before": s.nodes_before,
                    "nodes_after": s.nodes_after,
                    "details": s.details,
                    **({"duration_ms": round(s.duration_ms, 2)} if s.duration_ms is not None else {})
                }
                for s in self.steps
            ],
            "batch_classifications": self.batch_classifications,
            "results": self.results
        }
        if self.total_duration_ms is not None:
            result["total_duration_ms"] = round(self.total_duration_ms, 2)
        if self.classification_duration_ms is not None:
            result["classification_duration_ms"] = round(self.classification_duration_ms, 2)
        return result


class DenseXPathExecutor:
    """
    Executes XPath-like queries against an XML tree with semantic predicate matching.
    """
    
    TREE_PATH = Path(__file__).parent.parent / "storage" / "memory" / "tree_memory.xml"
    TRACE_DIR = Path(__file__).parent.parent / "reasoning_traces" / "traces"
    
    def __init__(self, classifier=None):
        """
        Initialize the executor.
        
        Args:
            classifier: Optional predicate classifier. If not provided, LLM classifier is used.
        """
        self._tree = None  # Lazy load
        self.classifier = classifier or LLMPredicateClassifier()
        self.step_counter = 0
    
    @property
    def tree(self) -> ET.Element:
        """Lazy load the XML tree."""
        if self._tree is None:
            tree = ET.parse(self.TREE_PATH)
            self._tree = tree.getroot()
        return self._tree
    
    def parse_query(self, query: str) -> list[PathSegment]:
        """
        Parse an XPath-like query into segments.
        
        Example: /Itinerary/Day[1]/POI[description =~ "jazz"]
        
        Returns list of PathSegment objects.
        """
        segments = []
        
        # Pattern matches:
        # - /Type
        # - /Type[index]
        # - /Type[description =~ "query"]
        pattern = r'/([A-Za-z]+)(?:\[(\d+)\])?(?:\[description\s*=~\s*"([^"]+)"\])?'
        
        for match in re.finditer(pattern, query):
            node_type = match.group(1)
            index = int(match.group(2)) if match.group(2) else None
            predicate = match.group(3)  # May be None
            
            segments.append(PathSegment(
                node_type=node_type,
                index=index,
                predicate=predicate
            ))
        
        return segments
    
    def _segment_to_string(self, segment: PathSegment) -> str:
        """Convert segment to string representation."""
        s = f"/{segment.node_type}"
        if segment.index is not None:
            s += f"[{segment.index}]"
        if segment.predicate:
            s += f'[description =~ "{segment.predicate}"]'
        return s
    
    def _build_tree_path(self, element: ET.Element, parent_path: str, sibling_index: int) -> str:
        """Build the tree path for an element."""
        return f"{parent_path}/{element.tag}[{sibling_index}]"
    
    def _get_children_by_type(self, parent: ET.Element, node_type: str) -> list[tuple[ET.Element, int]]:
        """Get children of a specific type with their sibling indices."""
        results = []
        type_count = 0
        for child in parent:
            if child.tag == node_type:
                type_count += 1
                results.append((child, type_count))
        return results
    
    def _element_to_subtree_text(self, element: ET.Element) -> str:
        """Convert element's children to text representation."""
        lines = []
        for i, child in enumerate(element, 1):
            name_elem = child.find("name")
            desc_elem = child.find("description")
            name = name_elem.text if name_elem is not None else "unnamed"
            desc = desc_elem.text[:100] if desc_elem is not None and desc_elem.text else ""
            lines.append(f"  [{i}] {child.tag}: {name}")
            if desc:
                lines.append(f"      {desc}")
        return "\n".join(lines)
    
    def _collect_predicates(self, segments: list[PathSegment]) -> list[tuple[int, str]]:
        """Collect all predicates with their segment indices (for bottom-up processing)."""
        predicates = []
        for i, seg in enumerate(segments):
            if seg.predicate:
                predicates.append((i, seg.predicate))
        return predicates
    
    def execute(self, query: str, save_trace: bool = True) -> tuple[list[NodeInfo], ExecutionTrace]:
        """
        Execute an XPath-like query against the tree.
        
        Algorithm:
        1. Parse query into segments
        2. Identify predicates from bottom to top
        3. For each predicate level (bottom-up):
           a. Collect candidate nodes at that level
           b. Batch classify with predicate
           c. Filter to matching nodes
        4. Return final matching nodes
        
        Args:
            query: XPath-like query string
            save_trace: Whether to save execution trace
            
        Returns:
            Tuple of (matching nodes, execution trace)
        """
        start_time = time.time()
        self.step_counter = 0
        
        # Initialize trace
        trace = ExecutionTrace(
            query=query,
            timestamp=datetime.now().isoformat(),
            segments=[]
        )
        
        # Parse query
        segments = self.parse_query(query)
        trace.segments = [self._segment_to_string(s) for s in segments]
        
        if not segments:
            trace.total_duration_ms = (time.time() - start_time) * 1000
            return [], trace
        
        # Collect predicates for bottom-up processing
        predicates = self._collect_predicates(segments)
        
        # Execute based on predicate structure
        if not predicates:
            # No predicates - just type/index matching
            results = self._execute_type_matching(segments, trace)
        elif len(predicates) == 1:
            # Single predicate - simple case
            results = self._execute_single_predicate(segments, predicates[0], trace)
        else:
            # Multiple predicates - bottom-up processing
            results = self._execute_multi_predicate(segments, predicates, trace)
        
        # Record results in trace
        trace.results = [
            {
                "tree_path": n.tree_path,
                "type": n.node_type,
                "name": n.name,
                "description": n.description[:100] if n.description else None
            }
            for n in results
        ]
        
        # Calculate total and classification durations
        trace.total_duration_ms = (time.time() - start_time) * 1000
        
        # Sum up classification durations from batch_classifications
        classification_time = sum(
            bc.get("duration_ms", 0) for bc in trace.batch_classifications
        )
        trace.classification_duration_ms = classification_time
        
        # Save trace
        if save_trace:
            self._save_trace(trace, query)
        
        return results, trace
    
    def _execute_type_matching(
        self,
        segments: list[PathSegment],
        trace: ExecutionTrace
    ) -> list[NodeInfo]:
        """Execute query with only type/index matching (no predicates)."""
        # Start from root
        current_nodes = [(self.tree, "/Itinerary", 1)]  # (element, path, sibling_idx)
        
        for seg in segments:
            if seg.node_type == "Itinerary":
                # Root - already matched
                continue
            
            next_nodes = []
            for elem, path, _ in current_nodes:
                children = self._get_children_by_type(elem, seg.node_type)
                
                for child, sibling_idx in children:
                    # Index filtering
                    if seg.index is not None and sibling_idx != seg.index:
                        continue
                    
                    child_path = self._build_tree_path(child, path, sibling_idx)
                    next_nodes.append((child, child_path, sibling_idx))
            
            self.step_counter += 1
            trace.steps.append(ExecutionStep(
                step_num=self.step_counter,
                action="type_match" if seg.index is None else "index_match",
                segment=self._segment_to_string(seg),
                nodes_before=len(current_nodes),
                nodes_after=len(next_nodes)
            ))
            
            current_nodes = next_nodes
        
        # Convert to NodeInfo
        return [
            NodeInfo.from_element(elem, path)
            for elem, path, _ in current_nodes
        ]
    
    def _execute_single_predicate(
        self,
        segments: list[PathSegment],
        predicate_info: tuple[int, str],
        trace: ExecutionTrace
    ) -> list[NodeInfo]:
        """Execute query with a single predicate."""
        pred_idx, predicate = predicate_info
        
        # First, do type matching up to predicate level
        current_nodes = [(self.tree, "/Itinerary", 1)]
        
        for i, seg in enumerate(segments):
            if seg.node_type == "Itinerary":
                continue
            
            next_nodes = []
            for elem, path, _ in current_nodes:
                children = self._get_children_by_type(elem, seg.node_type)
                
                for child, sibling_idx in children:
                    if seg.index is not None and sibling_idx != seg.index:
                        continue
                    child_path = self._build_tree_path(child, path, sibling_idx)
                    next_nodes.append((child, child_path, sibling_idx))
            
            self.step_counter += 1
            trace.steps.append(ExecutionStep(
                step_num=self.step_counter,
                action="type_match" if seg.index is None else "index_match",
                segment=self._segment_to_string(seg),
                nodes_before=len(current_nodes),
                nodes_after=len(next_nodes)
            ))
            
            current_nodes = next_nodes
            
            # If this segment has the predicate, do batch classification
            if i == pred_idx and seg.predicate:
                node_infos = [
                    NodeInfo.from_element(elem, path)
                    for elem, path, _ in current_nodes
                ]
                
                # Batch classify - get full result with details
                matching, batch_result = self.classifier.get_matching_nodes(node_infos, predicate)
                
                # Record batch classification in trace
                trace.batch_classifications.append(batch_result.to_dict())
                
                self.step_counter += 1
                trace.steps.append(ExecutionStep(
                    step_num=self.step_counter,
                    action="predicate_batch",
                    segment=f'[description =~ "{predicate}"]',
                    nodes_before=len(node_infos),
                    nodes_after=len(matching),
                    details={
                        "predicate": predicate,
                        "batch_classification_index": len(trace.batch_classifications) - 1,
                        "matched_paths": [n.tree_path for n in matching]
                    },
                    duration_ms=batch_result.duration_ms
                ))
                
                # Update current_nodes to only matching ones
                matching_paths = {n.tree_path for n in matching}
                current_nodes = [
                    (elem, path, idx) for elem, path, idx in current_nodes
                    if path in matching_paths
                ]
        
        return [
            NodeInfo.from_element(elem, path)
            for elem, path, _ in current_nodes
        ]
    
    def _execute_multi_predicate(
        self,
        segments: list[PathSegment],
        predicates: list[tuple[int, str]],
        trace: ExecutionTrace
    ) -> list[NodeInfo]:
        """
        Execute query with multiple predicates using bottom-up processing.
        
        For query like: /Itinerary/Day[description =~ "relaxing"]/Restaurant[description =~ "italian"]
        1. First process "italian" on Restaurant nodes
        2. Find which Day nodes have matching Restaurant children
        3. Batch classify those Day nodes with "relaxing" + subtree context
        """
        # Sort predicates by segment index (descending for bottom-up)
        predicates_sorted = sorted(predicates, key=lambda x: x[0], reverse=True)
        
        # Start with full type matching to get all leaf candidates
        all_nodes_by_level = self._collect_nodes_by_level(segments, trace)
        
        # Track which nodes pass at each level
        passing_nodes = {}  # level_idx -> set of tree_paths
        
        # Process predicates bottom-up
        for pred_idx, predicate in predicates_sorted:
            level_nodes = all_nodes_by_level.get(pred_idx, [])
            
            if not level_nodes:
                continue
            
            # If this is not the bottom level, add subtree context from passing children
            if pred_idx < max(p[0] for p in predicates):
                # Find child level with predicate
                child_pred_idx = min(p[0] for p in predicates if p[0] > pred_idx)
                child_passing = passing_nodes.get(child_pred_idx, set())
                
                # Filter to nodes that have passing children
                filtered_nodes = []
                for elem, path, idx in level_nodes:
                    has_passing_child = any(
                        cp.startswith(path + "/") for cp in child_passing
                    )
                    if has_passing_child:
                        node_info = NodeInfo.from_element(elem, path)
                        node_info.subtree_text = self._element_to_subtree_text(elem)
                        filtered_nodes.append(node_info)
                
                if not filtered_nodes:
                    return []
                
                # Batch classify with subtree context
                matching, batch_result = self.classifier.get_matching_nodes(filtered_nodes, predicate)
                
                # Record batch classification in trace
                trace.batch_classifications.append(batch_result.to_dict())
                
                self.step_counter += 1
                trace.steps.append(ExecutionStep(
                    step_num=self.step_counter,
                    action="predicate_batch_with_subtree",
                    segment=f'[description =~ "{predicate}"]',
                    nodes_before=len(filtered_nodes),
                    nodes_after=len(matching),
                    details={
                        "predicate": predicate,
                        "level": pred_idx,
                        "batch_classification_index": len(trace.batch_classifications) - 1,
                        "matched_paths": [n.tree_path for n in matching]
                    },
                    duration_ms=batch_result.duration_ms
                ))
                
                passing_nodes[pred_idx] = {n.tree_path for n in matching}
            
            else:
                # Bottom level - no subtree context needed
                node_infos = [
                    NodeInfo.from_element(elem, path)
                    for elem, path, idx in level_nodes
                ]
                
                matching, batch_result = self.classifier.get_matching_nodes(node_infos, predicate)
                
                # Record batch classification in trace
                trace.batch_classifications.append(batch_result.to_dict())
                
                self.step_counter += 1
                trace.steps.append(ExecutionStep(
                    step_num=self.step_counter,
                    action="predicate_batch",
                    segment=f'[description =~ "{predicate}"]',
                    nodes_before=len(node_infos),
                    nodes_after=len(matching),
                    details={
                        "predicate": predicate,
                        "level": pred_idx,
                        "batch_classification_index": len(trace.batch_classifications) - 1,
                        "matched_paths": [n.tree_path for n in matching]
                    },
                    duration_ms=batch_result.duration_ms
                ))
                
                passing_nodes[pred_idx] = {n.tree_path for n in matching}
        
        # Get the final results from the deepest predicate level
        deepest_pred_idx = max(p[0] for p in predicates)
        final_paths = passing_nodes.get(deepest_pred_idx, set())
        
        # Also filter by parent predicates
        for pred_idx, _ in predicates:
            if pred_idx < deepest_pred_idx:
                parent_paths = passing_nodes.get(pred_idx, set())
                final_paths = {
                    fp for fp in final_paths
                    if any(fp.startswith(pp + "/") for pp in parent_paths)
                }
        
        # Return matching leaf nodes
        leaf_level = all_nodes_by_level.get(deepest_pred_idx, [])
        return [
            NodeInfo.from_element(elem, path)
            for elem, path, idx in leaf_level
            if path in final_paths
        ]
    
    def _collect_nodes_by_level(
        self,
        segments: list[PathSegment],
        trace: ExecutionTrace
    ) -> dict[int, list[tuple[ET.Element, str, int]]]:
        """Collect all matching nodes at each level of the path."""
        nodes_by_level = {}
        current_nodes = [(self.tree, "/Itinerary", 1)]
        
        for i, seg in enumerate(segments):
            if seg.node_type == "Itinerary":
                nodes_by_level[i] = current_nodes
                continue
            
            next_nodes = []
            for elem, path, _ in current_nodes:
                children = self._get_children_by_type(elem, seg.node_type)
                
                for child, sibling_idx in children:
                    if seg.index is not None and sibling_idx != seg.index:
                        continue
                    child_path = self._build_tree_path(child, path, sibling_idx)
                    next_nodes.append((child, child_path, sibling_idx))
            
            self.step_counter += 1
            trace.steps.append(ExecutionStep(
                step_num=self.step_counter,
                action="type_match" if seg.index is None else "index_match",
                segment=self._segment_to_string(seg),
                nodes_before=len(current_nodes),
                nodes_after=len(next_nodes)
            ))
            
            nodes_by_level[i] = next_nodes
            current_nodes = next_nodes
        
        return nodes_by_level
    
    def _save_trace(self, trace: ExecutionTrace, query: str):
        """Save execution trace to file."""
        self.TRACE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename
        safe_query = re.sub(r'[/\[\]=~"\s]+', '_', query)[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_query}.json"
        filepath = self.TRACE_DIR / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"Trace saved to: {filepath}")


if __name__ == "__main__":
    executor = DenseXPathExecutor()
    
    test_queries = [
        "/Itinerary/Day[1]/Restaurant",
        "/Itinerary/Day/POI[description =~ \"jazz\"]",
        "/Itinerary/Day[1]/Restaurant[description =~ \"italian\"]",
    ]
    
    print("Testing Dense XPath Executor")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        results, trace = executor.execute(query)
        print(f"Found {len(results)} matching nodes:")
        for r in results:
            print(f"  - {r.tree_path}: {r.name}")
