"""
Dense XPath Executor - Main executor class that orchestrates query execution.

Paper Formalization - Hierarchical Retrieval:

Query Execution follows Eval(W, Q) where:
- W is a weighted node set: W ⊆ V × [0,1]
- Q = s₁/s₂/.../sₘ is a sequence of query steps
- Each step s = (axis, κ, ψ) has type κ and predicate ψ

Step Semantics - Step(W, s):
1. Structural expansion: Cand(v, s) = {u ∈ Axis(v) | κ(u) = κ}
2. Predicate scoring: Score(u, ψ) - recursive evaluation
3. Weight update: w_u = w · Score(u, ψ)

Predicate Types:
- ATOM: Local atomic predicate - Atom(u, φ) from attr(u)
- AND: Conjunction - Score(u, ψ₁) · Score(u, ψ₂)
- OR: Disjunction - max{Score(u, ψ₁), Score(u, ψ₂)}
- AGG_EXISTS: Existential aggregation - Agg∃ = max over children
- AGG_PREV: Prevalence aggregation - Aggprev = avg over children

Post-processing:
- TopK selection and threshold filtering
- LLM reasoning for final selection
"""

import logging
import yaml
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import time
from .predicate_scorer import PredicateScorer, get_scorer

from pipeline_execution.semantic_xpath_parsing import QueryParser
from pipeline_execution.semantic_xpath_parsing.parsing_models import (
    Axis,
    Index,
    IndexRange,
    NodeTestExpr,
    NodeTestAnd,
    NodeTestLeaf,
    NodeTestOr,
    PathExpr,
    Query,
    QueryStep,
    Step,
)
from pipeline_execution.semantic_xpath_parsing.predicate_ast import PredicateNode, AtomPredicate
from .execution_models import (
    MatchedNode, TraversalStep, ExecutionResult, NodeItem,
    StepContribution, NodeFusionTrace, ScoreFusionTrace, FinalFilteringTrace,
    ParsedQueryAST, DemoLoggerTrace
)
from pipeline_execution.semantic_xpath_util.node_utils import NodeUtils
from .index_handler import IndexHandler
from .predicate_handler import PredicateHandler
from utils.logger.demo_logging import get_demo_logger
from utils.logger.query_execution_logging import TraceWriter
from pipeline_execution.semantic_xpath_util.schema_loader import load_config, get_data_path, load_schema


logger = logging.getLogger(__name__)

# Small epsilon to avoid log(0)
EPSILON = 1e-9


class DenseXPathExecutor:
    """
    Executes XPath-like queries against an XML tree with semantic matching.
    
    Paper Formalization - Eval(W, Q):
    - Iterative step execution over Q = s₁/s₂/.../sₘ
    - Each step applies Step(W, s) for structural expansion and predicate scoring
    - Final score is product of step scores: Score(u, Q) = ∏ Score_i(u, ψ_i)
    
    Supports:
    - Type matching: /Itinerary/Day/POI (κ - node type)
    - Positional indexing: Day[1], POI[2], POI[-1], POI[1:3] (ι - positional constraint)
    - Atomic predicates: POI[atom(content =~ "museum")] (Atom(u, φ))
    - Compound predicates: POI[atom(...) AND atom(...)]
    - Hierarchical predicates: Day[agg_exists(POI[atom(...)])]
    - Global indexing: (/Itinerary/Day/POI)[5]
    - Multiple data files via schema configuration
    """
    
    def __init__(
        self, 
        scorer: PredicateScorer = None,
        scoring_method: str = None,
        top_k: int = None,
        score_threshold: float = None,
        config: dict = None,
        data_name: str = None,
        schema_name: str = None,
        tree_path: Path = None,
        traces_path: Path = None
    ):
        """
        Initialize the executor.
        
        Args:
            scorer: Predicate scorer implementation. If not provided, uses get_scorer().
            scoring_method: Scoring method ("llm" or "entailment"). 
                           If None, uses value from config.yaml.
            top_k: Number of top-scoring nodes to return. Defaults to config value.
            score_threshold: Minimum score threshold. Defaults to config value.
            config: Optional config dict. If not provided, loads from config.yaml.
            data_name: Name of the data file to use (e.g., "travel_memory_5day").
                      If None, uses active_data from config.yaml or schema's default.
            schema_name: Name of the schema to use. If None, uses active_schema from config.
            tree_path: Direct path to XML tree file. Overrides data_name/schema resolution.
            traces_path: Directory for trace files. If None, uses default traces folder.
        """
        if config is None:
            config = load_config()
        
        executor_config = config.get("xpath_executor", {})
        
        # Load configuration values
        self.scoring_method = scoring_method or executor_config.get("scoring_method", "llm")
        self.top_k = top_k if top_k is not None else executor_config.get("top_k", 5)
        self.score_threshold = score_threshold if score_threshold is not None else executor_config.get("score_threshold", 0.5)
        self.prune_top_k_at_each_step = executor_config.get("prune_top_k_at_each_step", None)
        
        # Store schema and data configuration
        self.schema_name = schema_name
        self.data_name = data_name
        
        # Resolve data file path using schema loader or override
        if tree_path:
            self._memory_path = Path(tree_path)
        else:
            self._memory_path = get_data_path(data_name=data_name, schema_name=schema_name)
        
        # Load schema for reference
        self._schema = load_schema(schema_name)
        
        # Initialize scorer
        scorer_traces_path = traces_path / "reasoning_traces" if traces_path else None
        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = get_scorer(
                method=self.scoring_method, 
                config=config,
                traces_path=scorer_traces_path
            )
        
        # Initialize components
        self.parser = QueryParser()
        
        # Create schema-aware NodeUtils instance for dynamic field lookup
        self._node_utils = NodeUtils(self._schema)
        
        # Pass full schema to predicate handler for children lookup
        self.predicate_handler = PredicateHandler(
            scorer=self.scorer,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            schema=self._schema
        )
        self.trace_writer = TraceWriter(
            traces_path=traces_path / "reasoning_traces" if traces_path else None
        )
        
        # Lazy-loaded tree
        self._tree = None
        self._root = None
    
    @property
    def memory_path(self) -> Path:
        """Get the path to the data file."""
        return self._memory_path
    
    @property
    def tree(self) -> ET.ElementTree:
        """Lazy load the XML tree."""
        if self._tree is None:
            self._tree = ET.parse(self._memory_path)
            self._root = self._tree.getroot()
        return self._tree
    
    @property
    def root(self) -> ET.Element:
        """Get the root element."""
        if self._root is None:
            _ = self.tree  # Trigger lazy load
        return self._root
    
    def _validate_predicate(
        self,
        predicate: PredicateNode,
        node_type: str,
        execution_log: List[str]
    ) -> None:
        """
        Validate predicate is a known AST node type.
        """
        if not isinstance(predicate, PredicateNode):
            raise ValueError(
                f"Invalid predicate type '{type(predicate).__name__}' on '{node_type}'. "
                f"Expected a PredicateNode subclass."
            )
    
    @property
    def root_type(self) -> str:
        """Get the root element's tag name."""
        return self.root.tag
    
    def execute(self, query: str) -> ExecutionResult:
        """
        Execute an XPath-like query against the tree.
        
        Implements the Semantic XPath execution semantics:
        1. Parse query into steps
        2. For each step: Expand → Filter → Select
        3. Accumulate log-odds across steps with semantic predicates
        4. Compute final scores with sigmoid
        5. Apply TopK and threshold filtering
        
        Args:
            query: XPath-like query string
            
        Returns:
            ExecutionResult with matched nodes and execution log
        """
        start_time = time.perf_counter()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        execution_log = []
        scoring_traces = []
        traversal_steps = []
        
        execution_log.append(f"[{timestamp}] Executing query: {query}")
        execution_log.append(f"Data file: {self._memory_path.name}")
        
        # Parse the query
        parsed_query = self.parser.parse(query)
        steps = parsed_query.path.steps
        global_index = parsed_query.global_index

        # Build AST representation for tracing
        parsed_ast = self._build_parsed_ast(steps, global_index)
        
        if global_index is not None:
            if global_index.to_end:
                execution_log.append(f"Global index range detected: [{global_index.start}:]")
            elif global_index.is_range:
                execution_log.append(f"Global index range detected: [{global_index.start}:{global_index.end}]")
            else:
                execution_log.append(f"Global index detected: {global_index.start}")
        
        execution_log.append(f"Parsed steps: {steps}")
        
        # Add AST tree visualization to log
        execution_log.append("")
        execution_log.append(parsed_ast.to_tree_string())
        
        # =====================================================================
        # Score Fusion: Track accumulated product per node
        # =====================================================================
        # Key: (node_id, path) to handle nodes that appear in multiple steps
        node_score_product: Dict[int, float] = defaultdict(lambda: 1.0)
        node_step_contributions: Dict[int, List[StepContribution]] = defaultdict(list)
        node_paths: Dict[int, str] = {}  # Track paths for trace
        
        # Initialize demo logger for tracking accumulated scores
        demo_logger = get_demo_logger()
        demo_logger.reset()  # Start fresh for this query
        
        # Execute traversal with path and score tracking
        root_type = self.root_type
        current_items: List[NodeItem] = [NodeItem(self.root, root_type, 1.0, 0)]
        
        for step_idx, step in enumerate(steps):
            execution_log.append(f"\n--- Step {step_idx + 1}: {step} ---")

            # Root step handling (simple leaf only)
            if self._is_root_step(step, root_type):
                current_items, traversal_step = self._handle_root_step_new(
                    current_items, step, step_idx, execution_log
                )
                if traversal_step:
                    traversal_steps.append(traversal_step)
                # Log root step for accumulated score visualization
                demo_logger.start_step(step_idx, str(step))
                accumulated_scores = demo_logger.get_accumulated_scores()
                for item in current_items:
                    previous_accumulated = accumulated_scores.get(item.path, 1.0)
                    node_name = self._node_utils.get_name(item.node)
                    demo_logger.log_node_score(
                        node_name=node_name,
                        node_path=item.path,
                        step_score=1.0,
                        previous_accumulated=previous_accumulated,
                    )
                    accumulated_scores[item.path] = previous_accumulated
                demo_logger.end_step()
                continue

            # Check if this step has predicates - start demo logging BEFORE scoring
            has_predicates = self._step_has_predicates(step)
            predicate_str = repr(step.test) if has_predicates else str(step)
            
            # Start tracking this step BEFORE scoring so parent contributions are captured
            demo_logger.start_step(step_idx, predicate_str)
            
            # Evaluate node test expression for this step (scoring happens here)
            next_items, step_scores, step_trace = self._apply_step_expr(
                current_items, step, step_idx, execution_log
            )
            traversal_steps.append(step_trace)

            if not next_items:
                execution_log.append("No matching nodes for step expression")
                demo_logger.end_step()  # End step even if no matches
                current_items = []
                break

            if has_predicates:
                scoring_traces.extend(step_trace.details.get("scoring_trace", []))
                
                for item in next_items:
                    node_id = id(item.node)
                    step_score = step_scores.get(node_id, 1.0)
                    step_score = max(EPSILON, min(1 - EPSILON, step_score))
                    
                    # Track previous accumulated score for visualization
                    previous_accumulated = node_score_product.get(node_id, 1.0)
                    
                    node_score_product[node_id] *= step_score
                    node_paths[node_id] = item.path
                    node_step_contributions[node_id].append(StepContribution(
                        step_index=step_idx,
                        predicate_str=predicate_str,
                        score=step_score
                    ))

            # Log to demo logger with accumulated score info for every step
            accumulated_scores = demo_logger.get_accumulated_scores()
            for item in next_items:
                node_id = id(item.node)
                step_score = step_scores.get(node_id, 1.0)
                step_score = max(EPSILON, min(1 - EPSILON, step_score))
                
                # Use node's own accumulated score if present, else inherit from nearest ancestor
                if item.path in accumulated_scores:
                    previous_accumulated = accumulated_scores[item.path]
                else:
                    previous_accumulated = self._get_parent_accumulated_score(
                        item.path, accumulated_scores
                    )
                
                # Log to demo logger
                node_name = self._node_utils.get_name(item.node)
                demo_logger.log_node_score(
                    node_name=node_name,
                    node_path=item.path,
                    step_score=step_score,
                    previous_accumulated=previous_accumulated,
                )
                
                # Update local view for subsequent lookups in this step
                accumulated_scores[item.path] = previous_accumulated * step_score
            
            demo_logger.end_step()

            current_items = next_items
            execution_log.append(f"After step: {len(current_items)} nodes remaining")
        
        # =====================================================================
        # Final Score Computation: Score Fusion (Product)
        # =====================================================================
        execution_log.append(f"\n=== Score Fusion ===")
        
        fusion_traces = []
        for item in current_items:
            node_id = id(item.node)
            accumulated_product = node_score_product.get(node_id, 1.0)
            
            # Use product directly, or keep default if no predicates
            if node_id in node_score_product:
                final_score = accumulated_product
            else:
                # No semantic predicates applied - keep default score
                final_score = item.score
            
            item.score = final_score
            
            # Build fusion trace
            fusion_traces.append(NodeFusionTrace(
                node_path=item.path,
                node_type=item.node.tag,
                step_contributions=node_step_contributions.get(node_id, []),
                accumulated_product=accumulated_product,
                final_score=final_score
            ))
            
            execution_log.append(
                f"  {item.path}: product={accumulated_product:.4f} → score={final_score:.4f}"
            )
        
        score_fusion_trace = ScoreFusionTrace(per_node_traces=fusion_traces)
        
        # =====================================================================
        # Parent Score Propagation: Inherit ancestor-level scores
        # =====================================================================
        # Build path -> fused_score map from ALL nodes scored in earlier steps.
        # This allows propagating parent scores (e.g., Day AGG scores) down to
        # child result nodes (e.g., POIs), giving a combined pipeline score.
        all_scored_paths: Dict[str, float] = {}
        for nid, npath in node_paths.items():
            all_scored_paths[npath] = node_score_product.get(nid, 1.0)
        
        # Track own_score and parent_score per path for frontend display
        own_score_map: Dict[str, float] = {}
        parent_score_map: Dict[str, float] = {}
        
        propagation_count = 0
        for idx, item in enumerate(current_items):
            own_score = item.score
            own_score_map[item.path] = own_score
            
            # Walk up ancestor paths and multiply their fused scores
            parent_product = 1.0
            path_parts = item.path.split(" > ")
            for i in range(len(path_parts) - 1, 0, -1):
                ancestor_path = " > ".join(path_parts[:i])
                if ancestor_path in all_scored_paths:
                    parent_product *= all_scored_paths[ancestor_path]
            
            parent_score_map[item.path] = parent_product
            
            if parent_product != 1.0:
                item.score *= parent_product
                fusion_traces[idx].final_score = item.score
                propagation_count += 1
                execution_log.append(
                    f"  Parent propagation: {item.path}: own={own_score:.4f} × parent={parent_product:.4f} → {item.score:.4f}"
                )
        
        if propagation_count > 0:
            execution_log.append(f"\n=== Parent Score Propagation: {propagation_count} nodes updated ===")
        
        # =====================================================================
        # Threshold Filtering (before global index)
        # =====================================================================
        execution_log.append(f"\n=== Threshold Filtering (threshold={self.score_threshold}) ===")
        
        before_threshold_count = len(current_items)
        
        # Apply threshold filter first
        current_items = [item for item in current_items if item.score >= self.score_threshold]
        
        # Sort by score descending
        current_items.sort(key=lambda x: x.score, reverse=True)
        
        execution_log.append(
            f"  Filtered: {before_threshold_count} → {len(current_items)} nodes"
        )
        
        # =====================================================================
        # Apply global index AFTER threshold filtering
        # =====================================================================
        # This ensures that e.g. "first museum" selects from nodes that actually
        # match "museum" (pass threshold), not the first node in document order.
        if global_index is not None and current_items:
            current_items, global_step = self._apply_global_index(
                current_items, global_index, len(steps), execution_log
            )
            traversal_steps.append(global_step)
        
        # =====================================================================
        # Final top_k limit
        # =====================================================================
        execution_log.append(f"\n=== Final Filtering (top_k={self.top_k}) ===")
        
        before_count = len(current_items)
        
        # Apply top_k limit
        current_items = current_items[:self.top_k]
        
        after_count = len(current_items)
        
        execution_log.append(
            f"  After top_k: {before_count} → {after_count} nodes"
        )
        
        final_filtering_trace = FinalFilteringTrace(
            before_filter_count=before_threshold_count,
            threshold=self.score_threshold,
            top_k=self.top_k,
            after_filter_count=after_count,
            filtered_nodes=[
                {
                    "path": item.path,
                    "score": item.score,  # Combined pipeline score (own × parent)
                    "type": item.node.tag,
                    "own_score": own_score_map.get(item.path, item.score),
                    "parent_score": parent_score_map.get(item.path, 1.0),
                }
                for item in current_items
            ]
        )
        
        # Convert to result format with full subtree
        matched_nodes = [
            self._node_utils.node_to_matched(item.node, item.path, item.score, item.context_trace) 
            for item in current_items
        ]
        
        # Calculate execution time
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        execution_log.append(f"\nFinal result: {len(matched_nodes)} nodes (sorted by score)")
        execution_log.append(f"⏱️  Query execution time: {execution_time_ms:.2f}ms")
        
        # Calculate aggregated token usage
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for trace in scoring_traces:
            if "token_usage" in trace and trace["token_usage"]:
                usage = trace["token_usage"]
                for k in total_token_usage:
                    total_token_usage[k] += usage.get(k, 0)
        
        # Build demo logger trace
        demo_trace = DemoLoggerTrace(
            step_traces=demo_logger.get_step_traces(),
            accumulated_scores=demo_logger.get_accumulated_scores(),
        )
        
        result = ExecutionResult(
            query=query,
            matched_nodes=matched_nodes,
            execution_log=execution_log,
            scoring_traces=scoring_traces,
            traversal_steps=traversal_steps,
            execution_time_ms=execution_time_ms,
            data_file=self._memory_path.name,
            score_fusion_trace=score_fusion_trace,
            final_filtering_trace=final_filtering_trace,
            token_usage=total_token_usage if total_token_usage["total_tokens"] > 0 else None,
            parsed_ast=parsed_ast,
            demo_logger_trace=demo_trace,
        )
        
        # Save traces
        self.trace_writer.save_traces(timestamp, result)
        
        return result
    
    def _items_to_info(
        self, 
        items: List[NodeItem]
    ) -> List[dict]:
        """Convert items to info dictionaries for tracing."""
        return [
            self._node_utils.to_info_dict(item.node, item.path, item.score)
            for item in items
        ]
    
    def _handle_root_step(
        self,
        current_items: List[NodeItem],
        step: QueryStep,
        step_idx: int,
        execution_log: List[str]
    ) -> Tuple[List[NodeItem], Optional[TraversalStep]]:
        """Handle the root step (works with any root type)."""
        root_type = self.root_type
        if current_items and current_items[0].node.tag == root_type:
            execution_log.append(f"Matched {root_type} root")
            traversal_step = TraversalStep(
                step_index=step_idx,
                step_query=str(step),
                nodes_before=[{"type": "root"}],
                nodes_after=[{"type": root_type, "path": root_type}],
                action="root_match",
                details={"matched": True}
            )
            return current_items, traversal_step
        else:
            execution_log.append(f"ERROR: Root is not {root_type}")
            return [], None

    def _is_root_step(self, step: Step, root_type: str) -> bool:
        if not isinstance(step.test, NodeTestLeaf):
            return False
        test = step.test.test
        if test.kind != "type" or test.name != root_type:
            return False
        if test.predicate or test.index:
            return False
        return True

    def _handle_root_step_new(
        self,
        current_items: List[NodeItem],
        step: Step,
        step_idx: int,
        execution_log: List[str]
    ) -> Tuple[List[NodeItem], Optional[TraversalStep]]:
        root_type = self.root_type
        if current_items and current_items[0].node.tag == root_type:
            execution_log.append(f"Matched {root_type} root")
            traversal_step = TraversalStep(
                step_index=step_idx,
                step_query=str(step),
                nodes_before=[{"type": "root"}],
                nodes_after=[{"type": root_type, "path": root_type}],
                action="root_match",
                details={"matched": True}
            )
            return current_items, traversal_step
        execution_log.append(f"ERROR: Root is not {root_type}")
        return [], None

    def _step_has_predicates(self, step: Step) -> bool:
        return any(step.test.get_all_predicates())

    def _get_parent_accumulated_score(
        self,
        node_path: str,
        accumulated_scores: Dict[str, float]
    ) -> float:
        """
        Find the nearest ancestor's accumulated score for a node path.
        Falls back to 1.0 if no ancestor is found.
        """
        path_parts = node_path.split(" > ")
        for i in range(len(path_parts) - 1, 0, -1):
            ancestor_path = " > ".join(path_parts[:i])
            if ancestor_path in accumulated_scores:
                return accumulated_scores[ancestor_path]
        return 1.0

    def _apply_step_expr(
        self,
        current_items: List[NodeItem],
        step: Step,
        step_idx: int,
        execution_log: List[str]
    ) -> Tuple[List[NodeItem], Dict[int, float], TraversalStep]:
        """
        Apply a full step: axis expansion + node test expression evaluation.
        """
        nodes_before = self._items_to_info(current_items)
        axis = step.axis
        parent_map = None
        if axis == Axis.DESC:
            parent_map = self._node_utils.build_parent_map(self.root)

        next_items: List[NodeItem] = []
        scores_map: Dict[int, float] = {}
        scoring_traces: List[Dict[str, Any]] = []

        # Count and log zero-score pruning
        zero_score_items = [item for item in current_items if item.score == 0.0]
        if zero_score_items:
            pruned_names = [self._node_utils.get_name(item.node) for item in zero_score_items]
            execution_log.append(
                f"  Zero-score pruning: skipping {len(zero_score_items)} items "
                f"({', '.join(pruned_names)})"
            )

        for group_id, item in enumerate(current_items):
            # Zero-score pruning: skip expanding children of nodes with score 0.0
            # since 0.0 × child_score = 0.0 always (mathematically safe, not a heuristic)
            if item.score == 0.0:
                continue
            items, local_scores, local_traces = self._eval_node_test_expr_context(
                item, group_id, axis, step.test, execution_log, parent_map
            )
            next_items.extend(items)
            for node_id, score in local_scores.items():
                scores_map[node_id] = max(scores_map.get(node_id, 0.0), score)
            scoring_traces.extend(local_traces)

        nodes_after = self._items_to_info(next_items)

        traversal_step = TraversalStep(
            step_index=step_idx,
            step_query=str(step),
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            action="node_test_expr",
            details={
                "axis": axis.value,
                "node_test_expr": step.test.to_dict(),
                "found_count": len(next_items),
                "scoring_trace": scoring_traces,
            }
        )
        
        # Apply Top-K semantic pruning at this intermediate step, if enabled
        # Only prune when the step has predicates — without predicates all nodes
        # score equally (1.0) and pruning would just keep the first K by insertion
        # order, arbitrarily dropping data.
        has_predicates = self._step_has_predicates(step)
        if self.prune_top_k_at_each_step and self.prune_top_k_at_each_step > 0 and has_predicates:
            original_len = len(next_items)
            
            # Prune out nodes below threshold
            valid_items = [item for item in next_items if scores_map.get(id(item.node), item.score) >= self.score_threshold]
            
            # Then group by parent and keep only Top-K
            groups = defaultdict(list)
            for item in valid_items:
                groups[item.parent_group_id].append(item)
                
            pruned_items = []
            for group_id, group_list in groups.items():
                group_list.sort(key=lambda x: scores_map.get(id(x.node), x.score), reverse=True)
                pruned_items.extend(group_list[:self.prune_top_k_at_each_step])
                
            if len(pruned_items) < original_len:
                execution_log.append(
                    f"  Top-K Pruning: Reduced from {original_len} to {len(pruned_items)} nodes (K={self.prune_top_k_at_each_step})"
                )
                
            next_items = pruned_items
            # Update traversal step details to reflect pruning
            traversal_step.details["pruned_count"] = original_len - len(next_items)
            traversal_step.nodes_after = self._items_to_info(next_items)
        
        return next_items, scores_map, traversal_step

    def _eval_node_test_expr_context(
        self,
        context_item: NodeItem,
        group_id: int,
        axis: Axis,
        expr: NodeTestExpr,
        execution_log: List[str],
        parent_map: Optional[Dict] = None
    ) -> Tuple[List[NodeItem], Dict[int, float], List[Dict[str, Any]]]:
        if isinstance(expr, NodeTestLeaf):
            return self._eval_node_test_leaf_context(
                context_item, group_id, axis, expr, execution_log, parent_map
            )

        if isinstance(expr, NodeTestOr):
            combined: Dict[Tuple[int, int], NodeItem] = {}
            traces: List[Dict[str, Any]] = []
            for child in expr.children:
                items, _, child_traces = self._eval_node_test_expr_context(
                    context_item, group_id, axis, child, execution_log, parent_map
                )
                traces.extend(child_traces)
                for item in items:
                    key = (id(item.node), item.parent_group_id)
                    if key not in combined or item.score > combined[key].score:
                        combined[key] = item
            items_list = list(combined.values())
            scores_map = _items_score_map(items_list)
            return items_list, scores_map, traces

        if isinstance(expr, NodeTestAnd):
            traces: List[Dict[str, Any]] = []
            child_results = []
            for child in expr.children:
                items, _, child_traces = self._eval_node_test_expr_context(
                    context_item, group_id, axis, child, execution_log, parent_map
                )
                traces.extend(child_traces)
                child_results.append({(id(item.node), item.parent_group_id): item for item in items})

            if not child_results:
                return [], {}, traces

            common_keys = set(child_results[0].keys())
            for mapping in child_results[1:]:
                common_keys &= set(mapping.keys())

            combined: Dict[Tuple[int, int], NodeItem] = {}
            for key in common_keys:
                first_item = child_results[0][key]
                min_score = min(mapping[key].score for mapping in child_results)
                combined[key] = NodeItem(
                    first_item.node,
                    first_item.path,
                    min_score,
                    first_item.parent_group_id
                )
            items_list = list(combined.values())
            scores_map = _items_score_map(items_list)
            return items_list, scores_map, traces

        return [], {}, []

    def _eval_node_test_leaf_context(
        self,
        context_item: NodeItem,
        group_id: int,
        axis: Axis,
        expr: NodeTestLeaf,
        execution_log: List[str],
        parent_map: Optional[Dict] = None
    ) -> Tuple[List[NodeItem], Dict[int, float], List[Dict[str, Any]]]:
        test = expr.test
        axis_val = axis.value if axis != Axis.NONE else "child"
        matches: List[ET.Element] = []
        next_items: List[NodeItem] = []

        if test.kind == "wildcard":
            if axis_val == "desc":
                matches = [
                    n for n in context_item.node.iter()
                    if n is not context_item.node and NodeUtils._is_structured_node(n)
                ]
            else:
                matches = [
                    child for child in context_item.node
                    if NodeUtils._is_structured_node(child)
                ]
        else:
            if axis_val == "desc":
                matches = [
                    n for n in context_item.node.iter(test.name)
                    if n is not context_item.node
                ]
            else:
                matches = list(context_item.node.findall(test.name))

        # Build NodeItems with path tracking
        if axis_val == "desc":
            if parent_map is None:
                parent_map = self._node_utils.build_parent_map(self.root)
            for child in matches:
                child_path = self._node_utils.get_path_from_ancestor_to_descendant(
                    context_item.node, child, context_item.path, parent_map
                )
                # Propagate context trace
                next_items.append(NodeItem(child, child_path, 1.0, group_id, list(context_item.context_trace)))
        else:
            for child in matches:
                child_name = self._node_utils.get_unique_child_name(child, context_item.node)
                path_sep = " > " if context_item.path else ""
                child_path = f"{context_item.path}{path_sep}{child_name}"
                # Propagate context trace
                next_items.append(NodeItem(child, child_path, 1.0, group_id, list(context_item.context_trace)))

        # Apply index within this context
        if test.index is not None:
            nodes_only = [item.node for item in next_items]
            indexed_nodes = IndexHandler.apply_index(nodes_only, test.index, execution_log)
            indexed_ids = {id(n) for n in indexed_nodes}
            next_items = [item for item in next_items if id(item.node) in indexed_ids]

        scores_map: Dict[int, float] = {} # Fix: _items_score_map not available in local scope? Use dict comprehension
        scores_map = {id(item.node): item.score for item in next_items} 
        traces: List[Dict[str, Any]] = []

        if test.predicate and next_items:
            nodes_only = [item.node for item in next_items]
            _, pred_scores, trace = self.predicate_handler.apply_semantic_predicate(
                nodes_only, test.predicate, execution_log
            )
            traces.append(trace)
            for item in next_items:
                score = pred_scores.get(id(item.node), item.score)
                item.score = score
                
                # Capture significant predicate match for reasoning context
                if score > 0.5:
                    # Provide a human readable description of the match
                    try:
                        pred_str = str(test.predicate)
                        # Truncate if extremely long to avoid cluttering prompt
                        if len(pred_str) > 200:
                            pred_str = pred_str[:197] + "..."
                    except:
                        pred_str = "predicate"
                        if hasattr(test.predicate, "value"):
                            pred_str = f'"{test.predicate.value}"'

                    context_entry = {
                        "node_name": item.path.split(" > ")[-1],
                        "predicate": pred_str,
                        "score": score,
                        "description": f"Matched {pred_str} on {item.path.split(' > ')[-1]} (Score: {score:.2f})"
                    }
                    item.context_trace.append(context_entry)

            scores_map = pred_scores

        return next_items, scores_map, traces
    
    def _apply_type_match(
        self,
        current_items: List[NodeItem],
        step: QueryStep,
        execution_log: List[str]
    ) -> List[NodeItem]:
        """
        Apply type matching to get children or descendants of specified type.
        
        Axis semantics:
        - child (default): Match only direct children using findall()
        - desc: Match all descendants at any depth using iter()
        - "*" (wildcard): Match all structured children regardless of type
        
        Each parent node's matches are assigned a unique parent_group_id,
        enabling local indexing like Day/POI[2] to select the 2nd POI
        within EACH Day rather than the global 2nd POI.
        
        For descendant axis (//), the full path through intermediate nodes
        is computed to ensure tree operations can find the correct location.
        """
        next_items = []
        axis = getattr(step, 'axis', 'child')  # Default to child for backward compatibility
        
        # For descendant axis, build parent map to trace full paths
        parent_map = None
        if axis == "desc":
            parent_map = self._node_utils.build_parent_map(self.root)
        
        for group_id, item in enumerate(current_items):
            if step.node_type in (".", "*"):
                # Wildcard: get all structured children regardless of type
                matches = [
                    child for child in item.node
                    if NodeUtils._is_structured_node(child)
                ]
                for child in matches:
                    child_name = self._node_utils.get_unique_child_name(child, item.node)
                    child_path = f"{item.path} > {child_name}"
                    next_items.append(NodeItem(child, child_path, item.score, group_id))
                    
            elif axis == "desc":
                # Descendant axis: find all descendants of this type at any depth
                # iter() returns self first, so we skip if it matches the type
                matches = [
                    n for n in item.node.iter(step.node_type) 
                    if n is not item.node
                ]
                for child in matches:
                    # Use full path tracing through intermediate nodes
                    child_path = self._node_utils.get_path_from_ancestor_to_descendant(
                        item.node, child, item.path, parent_map
                    )
                    next_items.append(NodeItem(child, child_path, item.score, group_id))
            else:
                # Child axis (default): find only direct children
                matches = list(item.node.findall(step.node_type))
                for child in matches:
                    child_name = self._node_utils.get_unique_child_name(child, item.node)
                    child_path = f"{item.path} > {child_name}"
                    next_items.append(NodeItem(child, child_path, item.score, group_id))
        
        # Build log message based on match type
        if step.node_type in (".", "*"):
            execution_log.append(
                f"Found {len(next_items)} children (wildcard) across {len(current_items)} parent(s)"
            )
        else:
            axis_desc = "descendants" if axis == "desc" else "children"
            execution_log.append(
                f"Found {len(next_items)} {step.node_type} {axis_desc} across {len(current_items)} parent(s)"
            )
        return next_items
    
    def _apply_predicate_step(
        self,
        items: List[NodeItem],
        step: QueryStep,
        step_idx: int,
        execution_log: List[str]
    ) -> Tuple[List[NodeItem], dict, TraversalStep, Dict[int, float]]:
        """
        Apply semantic predicate filtering.

        Returns:
            - filtered items (but no TopK/threshold - just the predicate)
            - scoring trace
            - traversal step
            - scores_map for Bayesian fusion
        """
        predicate_str = str(step.predicate) if step.predicate else step.predicate_str
        execution_log.append(f"Applying semantic predicate: {predicate_str}")

        nodes_before_pred = self._items_to_info(items)
        nodes_only = [item.node for item in items]

        # Use compound predicate if available, otherwise create from string
        if step.predicate:
            predicate = step.predicate
        else:
            # Legacy: convert string to AtomPredicate
            predicate = AtomPredicate(field="content", value=step.predicate_str)
        
        # Validate predicate uses required quantifier syntax
        self._validate_predicate(predicate, step.node_type, execution_log)

        # Apply predicate (no filtering - returns all nodes with scores)
        _, scores_map, trace = self.predicate_handler.apply_semantic_predicate(
            nodes_only, predicate, execution_log
        )
        
        # Update item scores from scores_map, preserving parent_group_id
        next_items = [
            NodeItem(
                item.node, 
                item.path, 
                scores_map.get(id(item.node), item.score), 
                item.parent_group_id
            )
            for item in items
        ]
        
        nodes_after_pred = self._items_to_info(next_items)
        
        traversal_step = TraversalStep(
            step_index=step_idx,
            step_query=str(step),
            nodes_before=nodes_before_pred,
            nodes_after=nodes_after_pred,
            action="semantic_predicate",
            details={
                "predicate": predicate_str,
                "predicate_type": type(predicate).__name__,
                "scoring_result": trace
            }
        )
        
        return next_items, trace, traversal_step, scores_map
    
    def _apply_index_step(
        self,
        items: List[NodeItem],
        step: QueryStep,
        step_idx: int,
        execution_log: List[str]
    ) -> Tuple[List[NodeItem], TraversalStep]:
        """
        Apply positional index filtering with LOCAL semantics.
        
        Groups items by parent_group_id and applies the index to each group
        separately. This enables Day/POI[2] to return the 2nd POI in EACH Day.
        """
        execution_log.append(f"Applying positional index: {step.index}")
        
        nodes_before_idx = self._items_to_info(items)
        
        # Group items by parent_group_id for local indexing
        groups = defaultdict(list)
        for item in items:
            groups[item.parent_group_id].append(item)
        
        # Apply index to each group separately
        next_items = []
        group_details = []
        
        for group_id in sorted(groups.keys()):
            group_items = groups[group_id]
            group_nodes = [item.node for item in group_items]
            
            # Apply index to this group's nodes
            indexed_nodes = IndexHandler.apply_index(group_nodes, step.index, execution_log)
            
            if indexed_nodes:
                indexed_set = set(id(n) for n in indexed_nodes)
                selected = [item for item in group_items if id(item.node) in indexed_set]
                next_items.extend(selected)
                group_details.append({
                    "group_id": group_id,
                    "group_size": len(group_items),
                    "selected_count": len(selected)
                })
        
        nodes_after_idx = self._items_to_info(next_items)
        
        traversal_step = TraversalStep(
            step_index=step_idx,
            step_query=str(step),
            nodes_before=nodes_before_idx,
            nodes_after=nodes_after_idx,
            action="positional_index",
            details={
                "index": step.index.to_dict(),
                "selected_count": len(next_items),
                "groups_processed": len(groups),
                "group_details": group_details
            }
        )
        
        return next_items, traversal_step
    
    def _apply_global_index(
        self,
        items: List[NodeItem],
        global_index: Index,
        total_steps: int,
        execution_log: List[str]
    ) -> Tuple[List[NodeItem], TraversalStep]:
        """
        Apply global index to final result set.
        
        Unlike local indexing, global indexing treats ALL nodes as a single
        flat list, regardless of parent_group_id.
        """
        execution_log.append(f"\nApplying global index: {global_index}")
        
        nodes_before_global = self._items_to_info(items)
        nodes_only = [item.node for item in items]
        
        indexed_nodes = IndexHandler.apply_index(nodes_only, global_index, execution_log)
        
        if indexed_nodes:
            indexed_set = set(id(n) for n in indexed_nodes)
            next_items = [item for item in items if id(item.node) in indexed_set]
        else:
            next_items = []
        
        nodes_after_global = self._items_to_info(next_items)
        
        traversal_step = TraversalStep(
            step_index=total_steps,
            step_query=f"global{global_index}",
            nodes_before=nodes_before_global,
            nodes_after=nodes_after_global,
            action="global_index",
            details={"index": global_index.to_dict()}
        )
        
        return next_items, traversal_step
    
    def _build_parsed_ast(
        self,
        steps: List[Step],
        global_index: Optional[Index]
    ) -> ParsedQueryAST:
        """
        Build a ParsedQueryAST object from parsed query steps.
        
        Converts Step objects and their node-test/predicate AST nodes into
        serializable dictionaries for logging and tracing.
        """
        ast_steps = []
        for step in steps:
            step_dict = {
                "axis": step.axis.value,
                "node_test_expr": step.test.to_dict(),
            }
            ast_steps.append(step_dict)

        global_idx_dict = global_index.to_dict() if global_index else None

        return ParsedQueryAST(steps=ast_steps, global_index=global_idx_dict)


def _items_score_map(items: List[NodeItem]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for item in items:
        node_id = id(item.node)
        scores[node_id] = max(scores.get(node_id, 0.0), item.score)
    return scores
