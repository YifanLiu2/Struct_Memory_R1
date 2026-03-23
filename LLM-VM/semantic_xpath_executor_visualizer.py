"""
Executor visualizer for Semantic XPath queries.

Prints:
  - Parsed AST tree (with predicate trees)
  - Per-step matched counts
  - Per-node score breakdowns (explicit predicate scoring)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from pipeline_execution.semantic_xpath_execution import DenseXPathExecutor
from pipeline_execution.semantic_xpath_execution.query_display import (
    canonicalize_query,
    canonicalize_parsed_ast_tree,
)
from pipeline_execution.semantic_xpath_parsing import QueryParser
from pipeline_execution.semantic_xpath_execution.predicate_scorer.base import (
    PredicateScorer,
    BatchScoringResult,
    ScoringResult,
)


class DummyScorer(PredicateScorer):
    """Simple keyword-based scorer for visualization."""

    def score_batch(self, nodes: List[Dict[str, Any]], predicate: str) -> BatchScoringResult:
        pred = (predicate or "").lower()
        results: List[ScoringResult] = []
        for node in nodes:
            desc = (node.get("description") or "").lower()
            score = 0.9 if pred and pred in desc else 0.1
            results.append(ScoringResult(
                node_id=node.get("id", ""),
                node_type=node.get("type", ""),
                node_description=node.get("description", ""),
                predicate=predicate,
                score=score,
                reasoning="keyword_match" if score > 0.5 else "no_match",
            ))
        return BatchScoringResult(predicate=predicate, results=results)


def _format_scoring_steps(scoring_steps: List[Dict[str, Any]], indent: int = 0) -> List[str]:
    if not scoring_steps:
        return []
    # Keep output brief: show the root computation (last step)
    return _format_scoring_step(scoring_steps[-1], indent)


def _format_scoring_step(step: Dict[str, Any], indent: int) -> List[str]:
    lines: List[str] = []
    pad = " " * indent
    step_type = step.get("type", "unknown")
    max_children = 3

    def _format_weight(value: Any) -> str:
        if isinstance(value, (int,)) or (isinstance(value, float) and value.is_integer()):
            return f"{int(value)}"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _format_best_match(best: Dict[str, Any], best_indent: int) -> List[str]:
        if not best:
            return []
        best_pad = " " * best_indent
        best_name = best.get("node_name", "?")
        best_type = best.get("node_type", "?")
        best_path = best.get("node_path", "")
        best_score = best.get("score", 0.0)
        out = [f"{best_pad}best match: {best_type} \"{best_name}\" (score={best_score:.2f})"]
        if best_path:
            out.append(f"{best_pad}  path: {best_path}")
        trace = best.get("trace", [])
        for line in _format_scoring_steps(trace, best_indent + 2):
            out.append(line)
        return out

    def _format_child_contributions(step_dict: Dict[str, Any], child_indent: int) -> List[str]:
        child_pad = " " * child_indent
        child_contribs = step_dict.get("child_contributions") or []
        if not child_contribs:
            child_results = step_dict.get("child_results", [])
            out: List[str] = []
            for child in child_results[:max_children]:
                out.append(
                    f"{child_pad}child score={child.get('score', 0.0):.2f} "
                    f"size={child.get('subtree_size', 0)}"
                )
            return out

        if step_type in ("agg_prev_weighted", "agg_prev"):
            key_fn = lambda c: c.get("weightedContribution", c.get("rawScore", 0.0))
        else:
            key_fn = lambda c: c.get("rawScore", 0.0)
        ordered = sorted(child_contribs, key=key_fn, reverse=True)
        out = [f"{child_pad}children (top {min(len(ordered), max_children)}):"]
        for child in ordered[:max_children]:
            name = child.get("childName", "?")
            ctype = child.get("childType", "?")
            score = child.get("rawScore", 0.0)
            size = child.get("subtreeSize", 0)
            weight = _format_weight(child.get("weight", 1.0))
            contrib = child.get("weightedContribution", score)
            best_flag = " best" if child.get("isBestMatch") else ""
            out.append(
                f"{child_pad}  - {ctype} \"{name}\" "
                f"score={score:.2f} size={size} weight={weight} "
                f"contrib={contrib:.2f}{best_flag}"
            )
        return out

    if step_type == "atom":
        cond = step.get("condition", {})
        field = cond.get("field", "content")
        op = cond.get("operator", "=~")
        value = cond.get("value", "")
        score = step.get("score", 0.0)
        lines.append(f"{pad}ATOM({field} {op} \"{value}\") = {score:.2f}")
        return lines

    if step_type == "and":
        result = step.get("result", 0.0)
        lines.append(f"{pad}AND result = {result:.2f}")
        for inner in step.get("inner_traces", []):
            if inner:
                lines.extend(_format_scoring_step(inner[-1], indent + 2))
        return lines

    if step_type == "or":
        result = step.get("result", 0.0)
        lines.append(f"{pad}OR result = {result:.2f}")
        for inner in step.get("inner_traces", []):
            if inner:
                lines.extend(_format_scoring_step(inner[-1], indent + 2))
        return lines

    if step_type == "not":
        result = step.get("result", 0.0)
        lines.append(f"{pad}NOT result = {result:.2f}")
        inner = step.get("inner_trace", [])
        if inner:
            lines.extend(_format_scoring_step(inner[-1], indent + 2))
        return lines

    if step_type in ("agg_exists_recursive", "agg_exists"):
        result = step.get("result", 0.0)
        lines.append(f"{pad}AGG_EXISTS result = {result:.2f}")
        lines.extend(_format_child_contributions(step, indent + 2))
        best_match = step.get("best_match_details")
        if best_match:
            lines.extend(_format_best_match(best_match, indent + 2))
        return lines

    if step_type in ("agg_prev_weighted", "agg_prev"):
        result = step.get("result", 0.0)
        lines.append(f"{pad}AGG_PREV result = {result:.2f}")
        weighted_sum = step.get("weighted_sum")
        total_weight = step.get("total_weight")
        if weighted_sum is not None and total_weight:
            lines.append(
                f"{pad}  weighted_sum={weighted_sum:.2f} total_weight={_format_weight(total_weight)}"
            )
        lines.extend(_format_child_contributions(step, indent + 2))
        best_match = step.get("best_match_details")
        if best_match:
            lines.extend(_format_best_match(best_match, indent + 2))
        return lines

    lines.append(f"{pad}{step_type}: {step}")
    return lines


def _format_index(index_dict: Dict[str, Any]) -> str:
    if not index_dict:
        return ""
    start = index_dict.get("start", "?")
    if index_dict.get("to_end"):
        return f"[{start}:]"
    if index_dict.get("end") is not None:
        return f"[{start}:{index_dict.get('end')}]"
    return f"[{start}]"


def _format_node_test_display(test: Dict[str, Any]) -> str:
    kind = test.get("kind")
    name = test.get("name")
    base = "*" if kind == "wildcard" else (name or "?")
    brackets: List[tuple[int, str]] = []

    if test.get("index"):
        idx_span = test.get("index", {}).get("span")
        order = idx_span.get("start", 1) if isinstance(idx_span, dict) else 1
        brackets.append((order, _format_index(test["index"])))

    for _, frag in sorted(brackets, key=lambda item: item[0]):
        base += frag
    return base


def _format_node_test_display_with_predicate(
    test: Dict[str, Any],
    predicate_str: str,
    predicate_dict: Optional[Dict[str, Any]] = None,
) -> str:
    kind = test.get("kind")
    name = test.get("name")
    base = "*" if kind == "wildcard" else (name or "?")
    brackets: List[tuple[int, str]] = []
    if predicate_str:
        pred_span = None
        if predicate_dict:
            pred_span = predicate_dict.get("span")
        order = pred_span.get("start", 0) if isinstance(pred_span, dict) else 0
        brackets.append((order, f"[{predicate_str}]"))
    if test.get("index"):
        idx_span = test.get("index", {}).get("span")
        order = idx_span.get("start", 1) if isinstance(idx_span, dict) else 1
        brackets.append((order, _format_index(test["index"])))
    for _, frag in sorted(brackets, key=lambda item: item[0]):
        base += frag
    return base


def _format_predicate_dict(pred: Dict[str, Any]) -> str:
    pred_type = pred.get("type") or pred.get("operator", "unknown")
    if pred_type == "atom":
        field = pred.get("field", "content")
        op = pred.get("operator", "=~")
        value = pred.get("value", "")
        return f'{field} {op} "{value}"'
    if pred_type == "AND":
        parts = [_format_predicate_dict(c) for c in pred.get("conditions", [])]
        return f"min({', '.join(parts)})"
    if pred_type == "OR":
        parts = [_format_predicate_dict(c) for c in pred.get("conditions", [])]
        return f"max({', '.join(parts)})"
    if pred_type == "NOT":
        inner = pred.get("condition")
        inner_str = _format_predicate_dict(inner) if inner else "?"
        return f"1 - ({inner_str})"
    if pred_type == "AGG_EXISTS":
        selector = pred.get("selector")
        inner = pred.get("child_predicate", {})
        if selector:
            axis = selector.get("axis", "child")
            axis_val = "child" if axis in (None, "none") else axis
            axis_prefix = "//" if axis_val == "desc" else ""
            selector_expr = _format_node_test_expr_with_predicate(
                selector.get("test", {}), inner, axis_prefix
            )
            return f"max({selector_expr})"
        child_type = pred.get("child_type", "*")
        axis = pred.get("child_axis", "child")
        axis_prefix = "//" if axis == "desc" else ""
        test = {
            "kind": "wildcard" if child_type in (None, "*") else "type",
            "name": None if child_type in (None, "*") else child_type,
        }
        selector_expr = axis_prefix + _format_node_test_with_predicate(test, inner)
        return f"max({selector_expr})"
    if pred_type == "AGG_PREV":
        selector = pred.get("selector")
        inner = pred.get("child_predicate", {})
        if selector:
            axis = selector.get("axis", "child")
            axis_val = "child" if axis in (None, "none") else axis
            axis_prefix = "//" if axis_val == "desc" else ""
            selector_expr = _format_node_test_expr_with_predicate(
                selector.get("test", {}), inner, axis_prefix
            )
            return f"avg({selector_expr})"
        child_type = pred.get("child_type", "*")
        axis = pred.get("child_axis", "child")
        axis_prefix = "//" if axis == "desc" else ""
        test = {
            "kind": "wildcard" if child_type in (None, "*") else "type",
            "name": None if child_type in (None, "*") else child_type,
        }
        selector_expr = axis_prefix + _format_node_test_with_predicate(test, inner)
        return f"avg({selector_expr})"
    return str(pred)


def _format_node_test_with_predicate(test: Dict[str, Any], predicate: Dict[str, Any]) -> str:
    if not predicate:
        return _format_node_test_display(test)
    pred_type = predicate.get("type") or predicate.get("operator", "unknown")
    if pred_type == "OR":
        parts = [
            _format_node_test_with_predicate(test, child)
            for child in predicate.get("conditions", [])
        ]
        return f"max({', '.join(parts)})"
    if pred_type == "AND":
        parts = [
            _format_node_test_with_predicate(test, child)
            for child in predicate.get("conditions", [])
        ]
        return f"min({', '.join(parts)})"
    predicate_str = _format_predicate_dict(predicate)
    return _format_node_test_display_with_predicate(test, predicate_str, predicate)


def _format_node_test_expr_with_predicate(
    expr: Dict[str, Any],
    predicate: Dict[str, Any],
    axis_prefix: str = "",
) -> str:
    if not expr:
        return "?"
    etype = expr.get("type")
    if etype == "leaf":
        test = expr.get("test", {})
        return axis_prefix + _format_node_test_with_predicate(test, predicate)
    if etype == "and":
        parts = [
            _format_node_test_expr_with_predicate(c, predicate, axis_prefix)
            for c in expr.get("children", [])
        ]
        return f"min({', '.join(parts)})"
    if etype == "or":
        parts = [
            _format_node_test_expr_with_predicate(c, predicate, axis_prefix)
            for c in expr.get("children", [])
        ]
        return f"max({', '.join(parts)})"
    return str(expr)


def _format_node_test_expr_display(expr: Dict[str, Any]) -> str:
    if not expr:
        return "?"
    etype = expr.get("type")
    if etype == "leaf":
        test = expr.get("test", {})
        predicate = test.get("predicate")
        if predicate:
            return _format_node_test_with_predicate(test, predicate)
        return _format_node_test_display(test)
    if etype == "and":
        parts = [_format_node_test_expr_display(c) for c in expr.get("children", [])]
        return f"min({', '.join(parts)})"
    if etype == "or":
        parts = [_format_node_test_expr_display(c) for c in expr.get("children", [])]
        return f"max({', '.join(parts)})"
    return str(expr)


def _format_step_display(step_trace: Dict[str, Any], step_index: int) -> str:
    details = step_trace.get("details", {})
    node_test_expr = details.get("node_test_expr")
    axis = details.get("axis")
    if node_test_expr:
        axis_val = "child" if axis in (None, "none") else axis
        axis_prefix = "//" if axis_val == "desc" else ""
        return f"{axis_prefix}{_format_node_test_expr_display(node_test_expr)}"
    return step_trace.get("step_query", f"step_{step_index}")


def _format_query_display(query: str) -> str:
    try:
        parser = QueryParser()
        parsed = parser.parse(query)
    except Exception:
        return query

    parts: List[str] = []
    for step in parsed.path.steps:
        axis_prefix = "//" if step.axis.value == "desc" else "/"
        step_str = _format_node_test_expr_display(step.test.to_dict())
        parts.append(f"{axis_prefix}{step_str}")
    path_str = "".join(parts) if parts else "/"
    if parsed.global_index:
        return f"({path_str}){parsed.global_index}"
    return path_str


def _format_parsed_ast_tree(parsed_ast: Any) -> str:
    steps = getattr(parsed_ast, "steps", [])
    global_index = getattr(parsed_ast, "global_index", None)
    lines = ["Query AST:"]
    for i, step in enumerate(steps):
        prefix = "├── " if i < len(steps) - 1 else "└── "
        axis = step.get("axis", "child")
        axis_val = "child" if axis in (None, "none") else axis
        axis_prefix = "//" if axis_val == "desc" else ""
        if "node_test_expr" in step:
            step_str = _format_node_test_expr_display(step["node_test_expr"])
            lines.append(f"{prefix}Step {i}: {axis_prefix}{step_str}")
        else:
            node_type = step.get("node_type", "?")
            lines.append(f"{prefix}Step {i}: {axis_prefix}{node_type}")
        if step.get("index"):
            lines.append(f"    │   index: {_format_index(step['index'])}")
    if global_index:
        lines.append(f"Global Index: {_format_index(global_index)}")
    return "\n".join(lines)


def _collect_leaf_predicates(expr: Dict[str, Any]) -> List[Optional[str]]:
    etype = expr.get("type")
    if etype == "leaf":
        test = expr.get("test", {})
        pred = test.get("predicate_str")
        return [pred] if pred else [None]
    if etype in ("and", "or"):
        preds: List[Optional[str]] = []
        for child in expr.get("children", []):
            preds.extend(_collect_leaf_predicates(child))
        return preds
    return []


def _print_step_results(step_trace: Dict[str, Any], step_index: int) -> None:
    nodes_after = step_trace.get("nodes_after", [])
    details = step_trace.get("details", {})
    node_test_expr = details.get("node_test_expr", {})
    expr_type = node_test_expr.get("type")
    leaf_predicates = _collect_leaf_predicates(node_test_expr)

    step_query = _format_step_display(step_trace, step_index)
    print(f"\nStep {step_index}: {step_query}")
    print(f"- matched: {len(nodes_after)}")
    if not nodes_after:
        return

    # Build predicate trace map: predicate_str -> node_id -> node_trace
    pred_trace_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for trace in details.get("scoring_trace", []):
        pred_label = trace.get("predicate", "predicate")
        for node_score in trace.get("node_scores", []):
            pred_trace_map.setdefault(pred_label, {})[node_score.get("node_id")] = node_score

    max_display = 3
    for node_info in nodes_after[:max_display]:
        node_id = node_info.get("node_id")
        name = node_info.get("name", "?")
        node_type = node_info.get("type", "?")
        score = node_info.get("score", 1.0)
        print(f"  - {node_type} \"{name}\" (score={score:.2f})")

        # If there are no predicates at all, note it.
        if not pred_trace_map:
            print("    predicate: <none>")
            continue

        # Compute and display combination rule if AND/OR
        if expr_type in ("and", "or") and leaf_predicates:
            leaf_scores: List[float] = []
            for pred in leaf_predicates:
                if pred is None:
                    leaf_scores.append(1.0)
                else:
                    node_trace = pred_trace_map.get(pred, {}).get(node_id)
                    leaf_scores.append(node_trace.get("final_score", 0.0) if node_trace else 0.0)
            if leaf_scores:
                if expr_type == "and":
                    combined = min(leaf_scores)
                    print(f"    AND combine = min({', '.join(f'{s:.2f}' for s in leaf_scores)}) = {combined:.2f}")
                else:
                    combined = max(leaf_scores)
                    print(f"    OR combine = max({', '.join(f'{s:.2f}' for s in leaf_scores)}) = {combined:.2f}")

        # Print predicate traces per predicate
        for pred_label, nodes_map in pred_trace_map.items():
            node_trace = nodes_map.get(node_id)
            if not node_trace:
                continue
            print(f"    predicate: {pred_label} (score={node_trace.get('final_score', 0.0):.2f})")
            for line in _format_scoring_steps(node_trace.get("scoring_steps", []), indent=6):
                print(f"{line}")

    if len(nodes_after) > max_display:
        print(f"  ... {len(nodes_after) - max_display} more nodes not shown")


def visualize_queries(tree_path: Path, queries: List[str]) -> None:
    executor = DenseXPathExecutor(
        scorer=DummyScorer(),
        scoring_method="dummy",
        top_k=1000,
        score_threshold=0.0,
        tree_path=tree_path,
    )

    for idx, query in enumerate(queries):
        print("\n" + "=" * 80)
        print(f"Query {idx + 1}")
        print(canonicalize_query(query))
        print("=" * 80)

        result = executor.execute(query)
        if result.parsed_ast:
            print("\n=== Query AST ===")
            print(canonicalize_parsed_ast_tree(result.parsed_ast))

        print("\n=== Step Results ===")
        for step in result.traversal_steps:
            if step.action in ("root_match", "node_test_expr"):
                _print_step_results(step.to_dict(), step.step_index)

        if result.final_filtering_trace:
            print("\n=== Final Filtering ===")
            print(
                f"before={result.final_filtering_trace.before_filter_count} "
                f"after={result.final_filtering_trace.after_filter_count}"
            )


def main() -> None:
    tree_path = Path("storage/memory/travel/travel_toronto_10day.xml")
    queries = [
        # 1) Global range + OR node tests + descendant axis + agg_exists + NOT
        (
            "(/Root/Itinerary_Version[-1]/Day[1]"
            "[agg_exists(//(POI OR Restaurant)[content =~ \"view\"]) "
            "AND not(atom(content =~ \"closed\"))]/"
            "(POI[1][atom(content =~ \"museum\")] OR "
            "Restaurant[1][atom(content =~ \"seafood\")]))[1:3]"
        ),
        # 2) Wildcard step + index range + AND/NOT in predicate
        (
            "/Root/Itinerary_Version[-1]/Day[1]/"
            "*[1:3][atom(content =~ \"morning\")]/"
            "POI[1:3][atom(content =~ \"coffee\") AND not(atom(content =~ \"decaf\"))]"
        ),
        # 3) NodeTestExpr AND between wildcard + typed node tests
        (
            "/Root/Itinerary_Version[-1]/Day[1]/"
            "(*[1:3][atom(content =~ \"tour\")] AND "
            "POI[1:3][atom(content =~ \"museum\")])"
        ),
        # 4) Recursive agg_exists -> agg_prev (1 level)
        (
            "/Root/Itinerary_Version[-1]/Day"
            "[agg_exists(//(POI OR Restaurant)"
            "[atom(content =~ \"park\")])]"
        ),
        # 5) Recursive agg_prev -> agg_exists -> agg_prev (2 levels) + global index
        (
            "(/Root/Itinerary_Version"
            "[agg_prev(Day"
            "[agg_prev((POI OR Restaurant)"
            "[atom(content =~ \"lunch\")])])]/Day[1:3])[1:3]"
        ),
        # 6) Predicate on Day + predicate on child POI
        (
            "/Root/Itinerary_Version[-1]/"
            "Day[1:3][atom(content =~ \"day\") AND agg_exists(//POI"
            "[atom(content =~ \"museum\")])]/"
            "POI[1:3][atom(content =~ \"art\") OR atom(content =~ \"gallery\")]"
        ),
        # 7) descendant axis + wildcard matching
        (
            "/Root/Itinerary_Version[-1]/"
            "//*[1:3][atom(content =~ \"view\")]/"
            "//POI[1:3][atom(content =~ \"park\")]"
        ),
    ]
    visualize_queries(tree_path, queries)


if __name__ == "__main__":
    main()
