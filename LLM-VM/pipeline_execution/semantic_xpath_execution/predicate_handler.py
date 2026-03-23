"""
Predicate Handler - Applies semantic predicate scoring to nodes.

Paper Formalization - Score(u, ψ):
The Score function is defined recursively over predicate structure:

  Score(u, ψ) = {
    Atom(u, φ)                           if ψ = φ (atomic predicate)
    min{Score(u, ψ₁), Score(u, ψ₂)}      if ψ = ψ₁ ∧ ψ₂ (conjunction)
    max{Score(u, ψ₁), Score(u, ψ₂)}      if ψ = ψ₁ ∨ ψ₂ (disjunction)
    1 - Score(u, ψ)                      if ψ = ¬ψ (negation)
  }

Atomic Predicate Evaluation - Atom(u, φ):
- Local: Atom(u, φ) evaluated from attr(u) - the node's own content
- Hierarchical: Atom(u, φ) = Agg({Atom(x, φ) | x ∈ Sφ(u)}) where Sφ(u) ⊆ Desc(u)

Aggregation Operators:
- Agg∃(A) = max A              (AGG_EXISTS - existential "at least one")
- Aggprev(A) = (1/|A|) ∑ A     (AGG_PREV - prevalence "on average")

Operator Mapping:
- ATOM: Local atomic predicate - Atom(u, φ) from attr(u)
- AND: Conjunction ψ₁ ∧ ψ₂ - min of scores
- OR: Disjunction ψ₁ ∨ ψ₂ - max of scores
- NOT: Negation ¬ψ - 1 minus score
- AGG_EXISTS: Hierarchical with Agg∃ - max over children
- AGG_PREV: Hierarchical with Aggprev - average over children

Batch Optimization:
- Collects all descriptions for the same atomic value across all nodes
- Makes one scorer call per unique atomic value
- Reduces N*M calls to just M calls (where M = unique atomic values)
"""

import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
from .predicate_scorer import PredicateScorer
from pipeline_execution.semantic_xpath_util.node_utils import NodeUtils
from utils.logger.demo_logging import get_demo_logger
from pipeline_execution.semantic_xpath_parsing.predicate_ast import (
    PredicateNode,
    AtomPredicate,
    AggPredicate,
    AggExistsPredicate,
    AggPrevPredicate,
    AndPredicate,
    OrPredicate,
    NotPredicate,
)
from pipeline_execution.semantic_xpath_parsing.parsing_models import (
    Axis,
    Index,
    NodeTest,
    NodeTestExpr,
    NodeTestLeaf,
    NodeTestAnd,
    NodeTestOr,
)


# Small epsilon to avoid log(0) and division by zero
EPSILON = 1e-9

# Type alias for scoring task: (node_id, desc_id, description_dict)
ScoringTask = Tuple[int, str, Dict[str, Any]]


class PredicateHandler:
    """
    Implements the recursive Score(u, ψ) function from paper formalization.
    
    Paper Formalization:
    - Score(u, ψ) recursively evaluates predicates over node u
    - Atom(u, φ) evaluates atomic predicates (local or hierarchical)
    
    Operator Scoring:
    - ATOM: Atom(u, φ) - local node content scoring
    - OR: max{Score(u, ψ₁), Score(u, ψ₂)} - disjunction
    - AND: min{Score(u, ψ₁), Score(u, ψ₂)} - conjunction
    - NOT: 1 - Score(u, ψ) - negation
    - AGG_EXISTS: Agg∃({Atom(x, φ) | x ∈ Sφ(u)}) = max - existential
    - AGG_PREV: Aggprev({Atom(x, φ) | x ∈ Sφ(u)}) = avg - prevalence
    """
    
    def __init__(
        self,
        scorer: PredicateScorer,
        top_k: int = 5,
        score_threshold: float = 0.5,
        schema: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the predicate handler.
        
        Args:
            scorer: PredicateScorer implementation (LLM, Entailment, or Cosine)
            top_k: Maximum number of nodes to return
            score_threshold: Minimum score threshold
            schema: Full schema dict with node definitions including 'children' field
        """
        self.scorer = scorer
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.schema = schema or {}
        
        # Extract node configs for quick lookup
        self._node_configs: Dict[str, Dict[str, Any]] = self.schema.get("nodes", {})
        
        # Create schema-aware NodeUtils instance for dynamic field lookup
        self._node_utils = NodeUtils(self.schema)
        
        # Cache for scores: (node_id, semantic_value) -> score
        self._score_cache: Dict[Tuple[int, str], float] = {}
        
        # Track node descriptions
        self._node_descriptions: Dict[int, List[str]] = {}
    
    def _get_allowed_children(self, node_type: str) -> List[str]:
        """Get the allowed child types for a node type from schema."""
        node_config = self._node_configs.get(node_type, {})
        return node_config.get("children", [])
    
    def _get_hierarchical_children(
        self, 
        node: ET.Element, 
        child_type: Optional[str] = None,
        axis: str = "child"
    ) -> List[ET.Element]:
        """
        Get hierarchical children or descendants of a node.
        
        Uses schema's 'children' field to distinguish structural children
        from the node's own fields (like name, description, etc.).
        
        Args:
            node: XML element
            child_type: Optional specific child type to filter for
            axis: "child" for direct children, "desc" for all descendants
            
        Returns:
            List of child/descendant elements that are recognized as structural
        """
        if axis == "desc":
            # Descendant axis: find all descendants of the specified type
            if child_type:
                # Get all descendants of the specified type (excluding self)
                return [n for n in node.iter(child_type) if n is not node]
            else:
                # Get all descendants (excluding self)
                # Filter to only include allowed children types recursively
                allowed_children = self._get_allowed_children(node.tag)
                if allowed_children:
                    result = []
                    for n in node.iter():
                        if n is not node and n.tag in self._get_all_structural_types():
                            result.append(n)
                    return result
                return list(node.iter())[1:]  # All descendants except self
        else:
            # Child axis (default): direct children only
            if child_type:
                direct = list(node.findall(child_type))
                if direct:
                    return direct
                
                # Schema-aware fallback: child_type not found as direct child.
                # Walk the schema to find intermediate containers that lead to
                # child_type.  e.g., Day -> Breakfast/Lunch/Dinner -> Meal
                allowed_children = self._get_allowed_children(node.tag)
                if allowed_children and child_type not in allowed_children:
                    results = []
                    for container_type in allowed_children:
                        container_children = self._get_allowed_children(container_type)
                        if child_type in container_children:
                            # Found a path: node -> container_type -> child_type
                            for container in node.findall(container_type):
                                results.extend(container.findall(child_type))
                    if results:
                        return results
                
                return []
            
            # Use schema's children definition for this node type
            allowed_children = self._get_allowed_children(node.tag)
            
            if allowed_children:
                return [child for child in node if child.tag in allowed_children]
            else:
                # Leaf node or no children defined - return empty
                return []

    def _apply_index_to_nodes(self, nodes: List[ET.Element], index: Optional[Index]) -> List[ET.Element]:
        if not index:
            return nodes
        from pipeline_execution.semantic_xpath_execution.index_handler import IndexHandler
        return IndexHandler.apply_index(nodes, index)  # type: ignore[arg-type]

    def _evaluate_node_test_expr(
        self,
        node: ET.Element,
        expr: NodeTestExpr,
        axis: Axis,
        execution_log: List[str]
    ) -> List[ET.Element]:
        """
        Evaluate a NodeTestExpr against a node to select evidence nodes.

        Predicates inside node tests are treated as filters using score_threshold.
        """
        if isinstance(expr, NodeTestLeaf):
            return self._evaluate_node_test_leaf(node, expr.test, axis, execution_log)

        if isinstance(expr, NodeTestOr):
            seen = set()
            result: List[ET.Element] = []
            for child in expr.children:
                for n in self._evaluate_node_test_expr(node, child, axis, execution_log):
                    nid = id(n)
                    if nid not in seen:
                        seen.add(nid)
                        result.append(n)
            return result

        if isinstance(expr, NodeTestAnd):
            child_lists = [
                self._evaluate_node_test_expr(node, child, axis, execution_log)
                for child in expr.children
            ]
            if not child_lists:
                return []
            # Intersection by node id, preserve order of first list
            common_ids = {id(n) for n in child_lists[0]}
            for lst in child_lists[1:]:
                common_ids &= {id(n) for n in lst}
            return [n for n in child_lists[0] if id(n) in common_ids]

        return []

    def _evaluate_node_test_leaf(
        self,
        node: ET.Element,
        test: NodeTest,
        axis: Axis,
        execution_log: List[str]
    ) -> List[ET.Element]:
        axis_str = axis.value if axis != Axis.NONE else "child"
        if test.kind == "wildcard":
            candidates = self._get_hierarchical_children(node, child_type=None, axis=axis_str)
        else:
            candidates = self._get_hierarchical_children(node, child_type=test.name, axis=axis_str)

        candidates = self._apply_index_to_nodes(candidates, test.index)

        if test.predicate:
            _, scores_map, _ = self.apply_semantic_predicate(candidates, test.predicate, execution_log)
            threshold = self.score_threshold
            return [n for n in candidates if scores_map.get(id(n), 0.0) >= threshold]
        return candidates
    
    def _get_all_structural_types(self) -> set:
        """Get all node types defined in the schema."""
        return set(self._node_configs.keys())
    
    def _recursive_subtree_score(
        self,
        node: ET.Element,
        child_predicate: PredicateNode,
        agg_operator: str,  # "EXISTS" or "PREV"
        trace_steps: List[Dict],
        execution_log: List[str]
    ) -> Tuple[float, int, Optional[Dict]]:
        """
        Recursively aggregate scores from node and its entire subtree.
        
        Bottom-up aggregation: scores leaf nodes first, then propagates up
        combining each node's own score with its descendants' scores.
        
        Args:
            node: XML element to score
            child_predicate: Predicate to evaluate at each node
            agg_operator: "EXISTS" (max, no weighting) or "PREV" (weighted avg)
            trace_steps: List to record scoring trace
            execution_log: Execution log for debugging
            
        Returns:
            Tuple of (aggregated_score, subtree_size, best_match_details) where:
            - aggregated_score: Combined score for this node and all descendants
            - subtree_size: Number of nodes in subtree (1 + sum of children sizes)
            - best_match_details: Scoring details of the node that contributed max score (for EXISTS)
        """
        # 1. Score this node's own content against the predicate
        # Capture detailed trace for this node's match
        node_match_trace = []
        own_score = self.score(node, child_predicate, node_match_trace, execution_log)
        
        # Structure the match details
        current_node_details = {
             "type": "node_match",
             "node_name": self._node_utils.get_name(node),
             "node_type": node.tag,
             "node_path": self._node_utils.get_path(node) if hasattr(self._node_utils, "get_path") else "",
             "score": own_score,
             "trace": node_match_trace
        }
        
        # 2. Get ALL hierarchical children (not filtered by type)
        children = self._get_hierarchical_children(node, child_type=None)
        
        if not children:
            # Leaf node - return own score and size=1
            return (own_score, 1, current_node_details)
        
        # 3. Recursively get (score, size) from all children
        child_results: List[Tuple[float, int, Optional[Dict]]] = []
        for child in children:
            child_score, child_size, child_details = self._recursive_subtree_score(
                child, child_predicate, agg_operator, trace_steps, execution_log
            )
            child_results.append((child_score, child_size, child_details))
        
        # 4. Calculate subtree size: 1 (self) + sum of children sizes
        total_children_size = sum(size for _, size, _ in child_results)
        subtree_size = 1 + total_children_size
        
        # 5. Aggregate based on operator
        best_match = current_node_details
        best_match_score = own_score
        
        if agg_operator == "EXISTS":
            # Max - no weighting (max is max)
            # Find max score and keeping track of which node/branch produced it
            max_score = own_score
            
            for c_score, c_size, c_details in child_results:
                if c_score > max_score:
                    max_score = c_score
                if c_details and c_details.get("score", -1) > best_match_score:
                    best_match_score = c_details["score"]
                    best_match = c_details
            
            result = max_score
        else:  # PREV - weighted average by subtree size
            # own_score has weight 1 (just this node)
            # each child has weight = its subtree size
            weighted_sum = own_score * 1
            total_weight = 1
            for child_score, child_size, _ in child_results:
                weighted_sum += child_score * child_size
                total_weight += child_size
            result = weighted_sum / total_weight
            for _, _, c_details in child_results:
                if c_details and c_details.get("score", -1) > best_match_score:
                    best_match_score = c_details["score"]
                    best_match = c_details
        
        return (result, subtree_size, best_match)
    
    # =========================================================================
    # Exact Attribute Matching (Fast Path) 
    # =========================================================================
    
    def _try_exact_attribute_match(
        self,
        nodes: List[ET.Element],
        predicate: PredicateNode,
        execution_log: List[str]
    ) -> Optional[Tuple[List[ET.Element], Dict[int, float], Dict[str, Any]]]:
        """
        Try to resolve a predicate using exact attribute matching instead of
        semantic scoring.
        
        For simple atom(content =~ "value") predicates, checks if the value
        matches any node's identifying attribute (e.g., Person/@name, Day/@index)
        using case-insensitive comparison.
        
        Returns:
            None if exact matching is not applicable (fall through to semantic scoring).
            Tuple of (nodes, scores_map, trace) if exact matches were found.
        """
        # Only applies to simple AtomPredicate
        if not isinstance(predicate, AtomPredicate):
            return None
        
        if not nodes:
            return None
        
        # Get the node type and its schema config
        node_type = nodes[0].tag
        node_config = self._node_configs.get(node_type, {})
        index_attr = node_config.get("index_attr")
        
        if not index_attr:
            return None
        
        # Try case-insensitive exact match against the identifying attribute
        query_value = predicate.value.strip().lower()
        exact_matches = []
        
        for node in nodes:
            attr_value = node.get(index_attr, "")
            if attr_value.strip().lower() == query_value:
                exact_matches.append(node)
        
        if not exact_matches:
            return None
        
        # Build scores: exact matches get 1.0, others get 0.0
        scores_map: Dict[int, float] = {}
        matched_ids = {id(n) for n in exact_matches}
        
        for node in nodes:
            scores_map[id(node)] = 1.0 if id(node) in matched_ids else 0.0
        
        # Build trace
        trace = {
            "predicate": str(predicate),
            "predicate_ast": predicate.to_dict(),
            "config": {
                "top_k": self.top_k,
                "score_threshold": self.score_threshold
            },
            "node_scores": [],
            "batch_scoring": {
                "exact_match": True,
                "matched_attribute": index_attr,
                "query_value": predicate.value,
                "matched_count": len(exact_matches),
            },
            "token_usage": None,
        }
        
        for idx, node in enumerate(nodes):
            node_name = self._node_utils.get_name(node)
            score = scores_map[id(node)]
            trace["node_scores"].append({
                "node_idx": idx,
                "node_id": id(node),
                "node_name": node_name,
                "node_type": node.tag,
                "final_score": score,
                "scoring_steps": [{
                    "type": "exact_attribute_match",
                    "attribute": index_attr,
                    "attribute_value": node.get(index_attr, ""),
                    "query_value": predicate.value,
                    "matched": id(node) in matched_ids,
                    "score": score,
                }]
            })
            
            execution_log.append(
                f"  Node {idx} ({node_name}): score = {score:.4f} (exact match)"
            )
        
        execution_log.append(
            f"  Exact attribute match on @{index_attr}: "
            f"{len(exact_matches)}/{len(nodes)} nodes matched"
        )
        
        trace["ranking"] = sorted(
            [{"idx": idx, "score": scores_map[id(node)], "name": self._node_utils.get_name(node)}
             for idx, node in enumerate(nodes)],
            key=lambda x: x["score"],
            reverse=True
        )
        
        return nodes, scores_map, trace
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    def apply_semantic_predicate(
        self, 
        nodes: List[ET.Element], 
        predicate: PredicateNode,
        execution_log: List[str] = None
    ) -> Tuple[List[ET.Element], Dict[int, float], Dict[str, Any]]:
        """
        Apply semantic predicate scoring to nodes.
        
        Uses batch optimization for efficient scoring.
        
        Args:
            nodes: List of XML elements to score
            predicate: CompoundPredicate AST
            execution_log: Optional list to append log messages
            
        Returns:
            - all nodes (no filtering - deferred to executor)
            - scores_map: dict mapping node id() to score
            - trace: detailed scoring trace
        """
        if execution_log is None:
            execution_log = []
        
        # Clear caches
        self._score_cache.clear()
        self._node_descriptions.clear()
        
        # Collect all atomic predicate values for logging
        atomic_values = predicate.get_all_atomic_values()
        execution_log.append(
            f"Scoring predicate: {predicate} (atomic values: {atomic_values})"
        )
        
        # Fast path: try exact attribute matching before expensive semantic scoring
        exact_result = self._try_exact_attribute_match(nodes, predicate, execution_log)
        if exact_result is not None:
            matched_nodes, scores_map, trace = exact_result
            return matched_nodes, scores_map, trace
        
        # Build trace structure
        trace = {
            "predicate": str(predicate),
            "predicate_ast": predicate.to_dict(),
            "config": {
                "top_k": self.top_k,
                "score_threshold": self.score_threshold
            },
            "node_scores": [],
            "batch_scoring": {}
        }
        
        # Phase 1: Collect all scoring tasks
        scoring_tasks = self._collect_scoring_tasks(nodes, predicate)
        
        execution_log.append(
            f"  Collected {sum(len(t) for t in scoring_tasks.values())} descriptions "
            f"for {len(scoring_tasks)} unique semantic value(s)"
        )
        
        # Phase 2: Batch score all semantics
        batch_stats = self._batch_score_semantics(scoring_tasks, trace)
        
        execution_log.append(
            f"  Made {batch_stats['total_scorer_calls']} scorer call(s) "
            f"for {batch_stats['total_descriptions_scored']} total descriptions"
        )
        
        # Phase 3: Compute final scores using recursive Score(u, ψ)
        scores_map: Dict[int, float] = {}
        
        for idx, node in enumerate(nodes):
            node_name = self._node_utils.get_name(node)
            node_trace = {
                "node_idx": idx,
                "node_id": id(node),
                "node_name": node_name,
                "node_type": node.tag,
                "scoring_steps": []
            }
            
            # Paper: Score(u, ψ) - recursive predicate evaluation
            score = self.score(
                node, predicate, node_trace["scoring_steps"], execution_log
            )
            
            scores_map[id(node)] = score
            node_trace["final_score"] = score
            trace["node_scores"].append(node_trace)
            
            execution_log.append(
                f"  Node {idx} ({node_name}): score = {score:.4f}"
            )
        
        # Add ranking info
        sorted_nodes = sorted(
            [(idx, scores_map[id(node)], self._node_utils.get_name(node)) 
             for idx, node in enumerate(nodes)],
            key=lambda x: x[1],
            reverse=True
        )
        
        trace["ranking"] = [
            {"idx": idx, "score": score, "name": name}
            for idx, score, name in sorted_nodes
        ]
        
        return nodes, scores_map, trace
    
    # =========================================================================
    # Phase 1: Collect Scoring Tasks
    # =========================================================================
    
    def _collect_scoring_tasks(
        self,
        nodes: List[ET.Element],
        predicate: PredicateNode
    ) -> Dict[str, List[ScoringTask]]:
        """Collect all scoring tasks grouped by semantic value."""
        tasks: Dict[str, List[ScoringTask]] = defaultdict(list)
        self._node_descriptions.clear()
        
        for node in nodes:
            self._collect_tasks_for_node(node, predicate, tasks)
        
        return tasks
    
    def _collect_tasks_for_node(
        self,
        node: ET.Element,
        predicate: PredicateNode,
        tasks: Dict[str, List[ScoringTask]]
    ):
        """
        Recursively collect scoring tasks for a node.
        
        Paper: Traverses predicate structure to find all Atom(u, φ) evaluations needed.
        """
        if isinstance(predicate, AtomPredicate):
            self._add_node_content_to_tasks(node, predicate.value, tasks)
        
        elif isinstance(predicate, (AndPredicate, OrPredicate)):
            for child in predicate.children:
                self._collect_tasks_for_node(node, child, tasks)
        
        elif isinstance(predicate, NotPredicate):
            if predicate.child:
                self._collect_tasks_for_node(node, predicate.child, tasks)
        
        elif isinstance(predicate, (AggExistsPredicate, AggPrevPredicate)):
            # Use _resolve_agg_children to respect path_prefix (e.g. "Dinner/Meal"
            # should only collect Dinner Meals, not all Breakfast+Lunch+Dinner Meals)
            children = self._resolve_agg_children(node, predicate.selector, [])
            for child in children:
                self._collect_subtree_tasks(child, predicate.inner, tasks)
    
    def _collect_subtree_tasks(
        self,
        node: ET.Element,
        predicate: PredicateNode,
        tasks: Dict[str, List[ScoringTask]]
    ):
        """
        Recursively collect scoring tasks for node and ALL descendants.
        
        Used by AGG_EXISTS/AGG_PREV to collect tasks for the entire subtree
        so that recursive bottom-up aggregation can score all nodes.
        """
        # Collect for this node
        self._collect_tasks_for_node(node, predicate, tasks)
        
        # Recurse into all hierarchical children (any type)
        children = self._get_hierarchical_children(node, child_type=None)
        for child in children:
            self._collect_subtree_tasks(child, predicate, tasks)
    
    def _add_node_content_to_tasks(
        self,
        node: ET.Element,
        semantic_value: str,
        tasks: Dict[str, List[ScoringTask]]
    ):
        """Add a node's content to scoring tasks.
        
        For container nodes (like Day), aggregates full content from all children.
        For leaf nodes, builds comprehensive content from ALL schema-defined fields
        (not just description) to enable temporal/cost-aware semantic matching.
        """
        node_id = id(node)
        
        # Skip if already added
        if any(t[0] == node_id for t in tasks.get(semantic_value, [])):
            return
        
        node_name = self._node_utils.get_name(node)
        
        # For container nodes, build comprehensive description from all children
        if NodeUtils._is_container_node(node):
            # Aggregate full content from all structured children
            parts = []
            
            # Include node's own identity (e.g. "Person: Father")
            if node_name:
                parts.append(f"{node.tag}: {node_name}")

            for child in node:
                if NodeUtils._is_structured_node(child):
                    # Use schema-aware field lookup for child name/desc
                    child_name = self._node_utils.get_field_value(child, "name")
                    child_desc = self._node_utils.get_field_value(child, "desc")
                    if child_name and child_desc:
                        parts.append(f"{child.tag}: {child_name} - {child_desc}")
                    elif child_name:
                        parts.append(f"{child.tag}: {child_name}")
                    elif child_desc:
                        parts.append(f"{child.tag}: {child_desc}")
            
            # Use aggregated description if available, otherwise fallback
            node_desc = "; ".join(parts) if parts else self._node_utils.get_description(node)
        else:
            # For leaf nodes, build content from ALL schema-defined fields
            node_desc = self._build_leaf_node_content(node)
        
        desc_ids = []
        
        if node_desc:
            desc_id = "main"
            desc_ids.append(desc_id)
            tasks[semantic_value].append((
                node_id,
                desc_id,
                {
                    "id": f"{node_id}_{desc_id}",
                    "type": node.tag,
                    "name": node_name,
                    "description": node_desc
                }
            ))
        
        self._node_descriptions[node_id] = desc_ids
    
    def _build_leaf_node_content(self, node: ET.Element) -> str:
        """
        Build comprehensive content string for a leaf node using schema-defined fields.
        
        Uses the schema's 'fields' list for the node type to include ALL relevant
        fields (name, time_block, description, expected_cost, etc.) in the content
        string. This enables semantic matching that considers temporal and cost context.
        
        Args:
            node: XML element (leaf node like POI, Restaurant)
            
        Returns:
            Content string with all field values, formatted as "field: value" pairs
        """
        node_type = node.tag
        node_config = self._node_configs.get(node_type, {})
        schema_fields = node_config.get("fields", [])
        
        parts = []
        
        if schema_fields:
            # Use schema-defined fields
            for field_name in schema_fields:
                elem = node.find(field_name)
                if elem is not None:
                    if len(elem) == 0 and elem.text:
                        # Simple text field
                        parts.append(f"{field_name}: {elem.text}")
                    elif len(elem) > 0:
                        # List field (like highlights)
                        items = [child.text for child in elem if child.text]
                        if items:
                            parts.append(f"{field_name}: {', '.join(items)}")
                else:
                    # Check for XML attribute (like @summary, @datetime)
                    attr_value = node.get(field_name)
                    if attr_value:
                        parts.append(f"{field_name}: {attr_value}")
        else:
            # Fallback: extract all text child elements
            for child in node:
                if len(child) == 0 and child.text:
                    parts.append(f"{child.tag}: {child.text}")
                elif len(child) > 0 and all(len(gc) == 0 for gc in child):
                    # Simple list
                    items = [gc.text for gc in child if gc.text]
                    if items:
                        parts.append(f"{child.tag}: {', '.join(items)}")
        
        return " | ".join(parts) if parts else self._node_utils.get_description(node)
    
    # =========================================================================
    # Phase 2: Batch Score Semantics
    # =========================================================================
    
    def _batch_score_semantics(
        self,
        scoring_tasks: Dict[str, List[ScoringTask]],
        trace: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call scorer once per unique semantic value."""
        batch_stats = {
            "semantic_values": list(scoring_tasks.keys()),
            "per_value_stats": [],
            "total_scorer_calls": 0,
            "total_descriptions_scored": 0
        }
        
        # Track total token usage
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for semantic_value, tasks in scoring_tasks.items():
            if not tasks:
                continue
            
            start_time = time.perf_counter()
            
            desc_dicts = [task[2] for task in tasks]
            batch_result = self.scorer.score_batch(desc_dicts, semantic_value)
            
            # Accumulate token usage if present
            if hasattr(batch_result, "token_usage") and batch_result.token_usage:
                for k in total_token_usage:
                    total_token_usage[k] += batch_result.token_usage.get(k, 0)
            
            call_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Store scores in cache
            for i, task in enumerate(tasks):
                node_id, desc_id, _ = task
                score = batch_result.results[i].score if i < len(batch_result.results) else 0
                score = max(EPSILON, min(1 - EPSILON, score))
                self._score_cache[(node_id, semantic_value)] = score
            
            batch_stats["per_value_stats"].append({
                "value": semantic_value,
                "num_descriptions": len(tasks),
                "call_time_ms": round(call_time_ms, 2)
            })
            batch_stats["total_scorer_calls"] += 1
            batch_stats["total_descriptions_scored"] += len(tasks)
        
        # Store accumulated token usage in trace
        trace["batch_scoring"] = batch_stats
        trace["token_usage"] = total_token_usage if total_token_usage["total_tokens"] > 0 else None
        
        return batch_stats
    
    # =========================================================================
    # Phase 3: Recursive Score Computation - Score(u, ψ)
    # =========================================================================
    
    def score(
        self,
        node: ET.Element,
        predicate: PredicateNode,
        trace_steps: List[Dict],
        execution_log: List[str]
    ) -> float:
        """
        Recursively score a predicate against a node.
        
        Paper Formalization - Score(u, ψ):
          Score(u, ψ) = {
            Atom(u, φ)                           if ψ = φ (atomic)
            min{Score(u, ψ₁), Score(u, ψ₂)}      if ψ = ψ₁ ∧ ψ₂ (AND)
            max{Score(u, ψ₁), Score(u, ψ₂)}      if ψ = ψ₁ ∨ ψ₂ (OR)
            1 - Score(u, ψ)                      if ψ = ¬ψ (NOT)
          }
        """
        if isinstance(predicate, AtomPredicate):
            return self._score_atom(node, predicate, trace_steps)
        if isinstance(predicate, OrPredicate):
            return self._score_or(node, predicate, trace_steps, execution_log)
        if isinstance(predicate, AndPredicate):
            return self._score_and(node, predicate, trace_steps, execution_log)
        if isinstance(predicate, NotPredicate):
            return self._score_not(node, predicate, trace_steps, execution_log)
        if isinstance(predicate, AggPredicate):
            agg_type = predicate.agg_type.lower()
            if agg_type == "prev":
                return self._score_agg_prev(node, predicate, trace_steps, execution_log)
            return self._score_agg_exists(node, predicate, trace_steps, execution_log)
        if isinstance(predicate, AggExistsPredicate):
            return self._score_agg_exists(node, predicate, trace_steps, execution_log)
        if isinstance(predicate, AggPrevPredicate):
            return self._score_agg_prev(node, predicate, trace_steps, execution_log)
        return 0
    
    def _score_atom(
        self,
        node: ET.Element,
        predicate: AtomPredicate,
        trace_steps: List[Dict]
    ) -> float:
        """Score atomic predicate - Atom(u, φ) from attr(u)."""
        node_id = id(node)
        cache_key = (node_id, predicate.value)
        score = self._score_cache.get(cache_key, 0)
        
        trace_steps.append({
            "type": "atom",
            "condition": predicate.to_dict(),
            "score": score,
            "note": "Atom(u, φ) - local node content from attr(u)"
        })
        
        return score
    
    def _score_or(
        self,
        node: ET.Element,
        predicate: OrPredicate,
        trace_steps: List[Dict],
        execution_log: List[str]
    ) -> float:
        """Score disjunction: max{Score(u, ψ_j)}"""
        trace_steps_list = []
        child_scores = []
        
        for child in predicate.children:
            inner_trace: List[Dict] = []
            s = self.score(node, child, inner_trace, execution_log)
            child_scores.append(s)
            trace_steps_list.append(inner_trace)
        
        result = max(child_scores) if child_scores else 0
        result = max(EPSILON, min(1 - EPSILON, result))
        
        trace_steps.append({
            "type": "or",
            "formula": "max{Score(u, ψ_j)}",
            "child_scores": child_scores,
            "inner_traces": trace_steps_list,
            "result": result
        })
        
        return result
    
    def _score_and(
        self,
        node: ET.Element,
        predicate: AndPredicate,
        trace_steps: List[Dict],
        execution_log: List[str]
    ) -> float:
        """Score conjunction: min{Score(u, ψ_j)}"""
        trace_steps_list = []
        child_scores = []
        
        for child in predicate.children:
            inner_trace: List[Dict] = []
            s = self.score(node, child, inner_trace, execution_log)
            child_scores.append(s)
            trace_steps_list.append(inner_trace)
        
        result = min(child_scores) if child_scores else 0
        result = max(EPSILON, min(1 - EPSILON, result))
        
        trace_steps.append({
            "type": "and",
            "formula": "min{Score(u, ψ_j)}",
            "child_scores": child_scores,
            "inner_traces": trace_steps_list,
            "result": result
        })
        
        return result
    
    def _score_not(
        self,
        node: ET.Element,
        predicate: NotPredicate,
        trace_steps: List[Dict],
        execution_log: List[str]
    ) -> float:
        """Score negation: 1 - Score(u, ψ)"""
        inner_trace: List[Dict] = []
        inner_score = self.score(node, predicate.child, inner_trace, execution_log) if predicate.child else 0
        
        result = 1 - inner_score
        result = max(EPSILON, min(1 - EPSILON, result))
        
        trace_steps.append({
            "type": "not",
            "formula": "1 - Score(u, ψ)",
            "inner_score": inner_score,
            "inner_trace": inner_trace,
            "result": result
        })
        
        return result
    
    def _resolve_agg_children(
        self,
        node: ET.Element,
        selector,
        execution_log: List[str]
    ) -> List[ET.Element]:
        """
        Navigate through selector's path_prefix and evaluate the final test.
        
        For a selector like Breakfast/Meal (path_prefix=["Breakfast"], test=Meal),
        first finds all Breakfast children, then finds Meal children within them.
        """
        # Start with the given node, navigate through path_prefix steps
        current_nodes = [node]
        for step_name in (selector.path_prefix or []):
            next_nodes = []
            for n in current_nodes:
                next_nodes.extend(child for child in n if child.tag == step_name)
            current_nodes = next_nodes
            if not current_nodes:
                return []
        
        # Now evaluate the final selector test against all resolved nodes
        children = []
        for n in current_nodes:
            children.extend(
                self._evaluate_node_test_expr(n, selector.test, selector.axis, execution_log)
            )
        return children
    
    def _score_agg_exists(
        self,
        node: ET.Element,
        predicate: AggPredicate,
        trace_steps: List[Dict],
        execution_log: List[str]
    ) -> float:
        """Hierarchical existential aggregation: Agg∃(A) = max(subtree scores)"""
        selector = predicate.selector
        children = self._resolve_agg_children(node, selector, execution_log)
        
        parent_name = self._node_utils.get_name(node)
        parent_path = self._node_utils.get_path(node) if hasattr(self._node_utils, 'get_path') else ""

        if not children:
            trace_steps.append({
                "type": "agg_exists",
                "selector": selector.to_dict(),
                "num_children": 0,
                "note": "Sφ(u) is empty - no evidence nodes found",
                "result": 0
            })
            return 0

        child_results: List[Tuple[float, int, Optional[Dict], ET.Element]] = []
        for child in children:
            score, size, details = self._recursive_subtree_score(
                child, predicate.inner, "EXISTS", trace_steps, execution_log
            )
            child_results.append((score, size, details, child))

        max_score = -1.0
        best_details = None
        best_child = None
        for s, _, d, c in child_results:
            if s > max_score:
                max_score = s
                best_details = d
                best_child = c

        result = max(EPSILON, min(1 - EPSILON, max_score)) if child_results else 0

        # Build detailed child contributions for demo logging
        child_contributions = []
        for score, size, details, child in child_results:
            child_name = self._node_utils.get_name(child)
            child_path = self._node_utils.get_path(child) if hasattr(self._node_utils, 'get_path') else ""
            is_best = (child == best_child)
            child_contributions.append({
                "childName": child_name,
                "childPath": child_path,
                "childType": child.tag,
                "rawScore": score,
                "weight": 1.0,  # No weighting for EXISTS
                "weightedContribution": score,
                "subtreeSize": size,
                "isBestMatch": is_best,
            })
        
        # Log to demo logger
        demo_logger = get_demo_logger()
        demo_logger.log_parent_contribution(
            parent_name=parent_name,
            parent_path=parent_path,
            parent_type=node.tag,
            predicate_type="agg_exists",
            formula="Agg∃(A) = max(recursive_subtree_scores)",
            child_contributions=child_contributions,
            aggregated_score=result,
        )

        trace_steps.append({
            "type": "agg_exists_recursive",
            "formula": "Agg∃(A) = max(recursive_subtree_scores)",
            "selector": selector.to_dict(),
            "num_children": len(children),
            "child_results": [{"score": s, "subtree_size": sz} for s, sz, _, _ in child_results],
            "child_contributions": child_contributions,  # Include detailed contributions
            "best_match_details": best_details,
            "result": result
        })

        return result
    
    def _score_agg_prev(
        self,
        node: ET.Element,
        predicate: AggPredicate,
        trace_steps: List[Dict],
        execution_log: List[str]
    ) -> float:
        """Hierarchical prevalence aggregation: Aggprev = weighted avg by subtree size"""
        selector = predicate.selector
        children = self._resolve_agg_children(node, selector, execution_log)
        
        parent_name = self._node_utils.get_name(node)
        parent_path = self._node_utils.get_path(node) if hasattr(self._node_utils, 'get_path') else ""

        if not children:
            trace_steps.append({
                "type": "agg_prev",
                "selector": selector.to_dict(),
                "num_children": 0,
                "note": "Sφ(u) is empty - no evidence nodes found",
                "result": 0
            })
            return 0

        child_results: List[Tuple[float, int, Optional[Dict], ET.Element]] = []
        for child in children:
            score, size, details = self._recursive_subtree_score(
                child, predicate.inner, "PREV", trace_steps, execution_log
            )
            child_results.append((score, size, details, child))

        weighted_sum = sum(score * size for score, size, _, _ in child_results)
        total_weight = sum(size for _, size, _, _ in child_results)
        result = weighted_sum / total_weight if total_weight > 0 else 0
        result = max(EPSILON, min(1 - EPSILON, result))

        # Build detailed child contributions for demo logging
        child_contributions = []
        best_details = None
        best_details_score = -1.0
        for score, size, details, child in child_results:
            child_name = self._node_utils.get_name(child)
            child_path = self._node_utils.get_path(child) if hasattr(self._node_utils, 'get_path') else ""
            weighted_contribution = score * size
            child_contributions.append({
                "childName": child_name,
                "childPath": child_path,
                "childType": child.tag,
                "rawScore": score,
                "weight": size,  # Subtree size as weight for PREV
                "weightedContribution": weighted_contribution,
                "subtreeSize": size,
            })
            if details and details.get("score", -1) > best_details_score:
                best_details_score = details["score"]
                best_details = details
        
        # Log to demo logger
        demo_logger = get_demo_logger()
        demo_logger.log_parent_contribution(
            parent_name=parent_name,
            parent_path=parent_path,
            parent_type=node.tag,
            predicate_type="agg_prev",
            formula=f"Aggprev(A) = {weighted_sum:.4f} / {total_weight} = {result:.4f}",
            child_contributions=child_contributions,
            aggregated_score=result,
        )

        trace_steps.append({
            "type": "agg_prev_weighted",
            "formula": "Aggprev(A) = Σ(score_i × size_i) / Σ(size_i)",
            "selector": selector.to_dict(),
            "num_children": len(children),
            "child_results": [{"score": s, "subtree_size": sz} for s, sz, _, _ in child_results],
            "child_contributions": child_contributions,  # Include detailed contributions
            "best_match_details": best_details,
            "weighted_sum": weighted_sum,
            "total_weight": total_weight,
            "result": result
        })

        return result
