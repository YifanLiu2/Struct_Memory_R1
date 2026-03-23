"""
Query display helpers for canonical, simplified rendering.

Outputs a simplified operation form for display/logging while preserving
execution grammar elsewhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pipeline_execution.semantic_xpath_parsing import QueryParser


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


def canonicalize_node_test_expr(expr: Dict[str, Any]) -> str:
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
        parts = [canonicalize_node_test_expr(c) for c in expr.get("children", [])]
        return f"min({', '.join(parts)})"
    if etype == "or":
        parts = [canonicalize_node_test_expr(c) for c in expr.get("children", [])]
        return f"max({', '.join(parts)})"
    return str(expr)


def canonicalize_query(query: str) -> str:
    try:
        parser = QueryParser()
        parsed = parser.parse(query)
    except Exception:
        return query

    parts: List[str] = []
    for step in parsed.path.steps:
        axis_prefix = "//" if step.axis.value == "desc" else "/"
        step_str = canonicalize_node_test_expr(step.test.to_dict())
        parts.append(f"{axis_prefix}{step_str}")
    path_str = "".join(parts) if parts else "/"
    if parsed.global_index:
        return f"({path_str}){parsed.global_index}"
    return path_str


def canonicalize_parsed_ast_tree(parsed_ast: Any) -> str:
    steps = getattr(parsed_ast, "steps", None)
    global_index = getattr(parsed_ast, "global_index", None)
    if steps is None and isinstance(parsed_ast, dict):
        steps = parsed_ast.get("steps", [])
        global_index = parsed_ast.get("global_index")
    if steps is None:
        return ""
    lines = ["Query AST:"]
    for i, step in enumerate(steps):
        prefix = "├── " if i < len(steps) - 1 else "└── "
        axis = step.get("axis", "child")
        axis_val = "child" if axis in (None, "none") else axis
        axis_prefix = "//" if axis_val == "desc" else ""
        if "node_test_expr" in step:
            step_str = canonicalize_node_test_expr(step["node_test_expr"])
            lines.append(f"{prefix}Step {i}: {axis_prefix}{step_str}")
        else:
            node_type = step.get("node_type", "?")
            lines.append(f"{prefix}Step {i}: {axis_prefix}{node_type}")
        if step.get("index"):
            lines.append(f"    │   index: {_format_index(step['index'])}")
    if global_index:
        lines.append(f"Global Index: {_format_index(global_index)}")
    return "\n".join(lines)
