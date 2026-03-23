"""
Predicate AST Node Types - Typed hierarchy for semantic XPath predicates.

Paper Formalization - Score(u, ψ):
    AtomPredicate:       Atom(u, φ) - local node content scoring
    AndPredicate:        ψ₁ ∧ ψ₂   - min{Score(u, ψ₁), Score(u, ψ₂)}
    OrPredicate:         ψ₁ ∨ ψ₂   - max{Score(u, ψ₁), Score(u, ψ₂)}
    NotPredicate:        ¬ψ         - 1 - Score(u, ψ)
    AggExistsPredicate:  Agg∃       - max over children/descendants
    AggPrevPredicate:    Aggprev    - average over children/descendants
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterable, Tuple

from pipeline_execution.semantic_xpath_parsing.parsing_models import (
    Axis,
    EvidenceSelector,
    NodeTest,
    NodeTestExpr,
    NodeTestLeaf,
    SourceSpan,
)


class PredicateNode(ABC):
    """Base class for all predicate AST nodes."""

    @abstractmethod
    def get_all_atomic_values(self) -> List[str]:
        """Extract all atomic predicate values for batch scoring."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for tracing."""
        ...

    # Backward-compat alias used by some callers
    def get_all_semantic_values(self) -> List[str]:
        return self.get_all_atomic_values()


# =============================================================================
# Base-case predicates
# =============================================================================

@dataclass
class AtomPredicate(PredicateNode):
    """
    atom(field =~ "value") - local semantic match on a node's content.

    Paper: Atom(u, φ) evaluated from attr(u).
    """
    field: str   # "content" for aggregated content, or specific field name
    value: str   # The semantic query value, e.g., "museum"
    operator: str = "=~"
    span: Optional[SourceSpan] = None

    def get_all_atomic_values(self) -> List[str]:
        return [self.value]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": "atom",
            "field": self.field,
            "value": self.value,
            "operator": self.operator,
        }
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def __repr__(self) -> str:
        return f'atom({self.field} {self.operator} "{self.value}")'


@dataclass
class AggPredicate(PredicateNode):
    """
    Aggregate predicate with an evidence selector and inner predicate.
    """
    agg_type: str  # "exists" | "prev" | "or"
    selector: EvidenceSelector
    inner: PredicateNode
    span: Optional[SourceSpan] = None

    def get_all_atomic_values(self) -> List[str]:
        values: List[str] = []
        # Include values from evidence selector predicates
        for pred in self.selector.test.get_all_predicates():
            if pred:
                values.extend(pred.get_all_atomic_values())
        values.extend(self.inner.get_all_atomic_values())
        return values

    def to_dict(self) -> Dict[str, Any]:
        op = "AGG_PREV" if self.agg_type.lower() == "prev" else "AGG_EXISTS"
        result: Dict[str, Any] = {
            "operator": op,
            "selector": self.selector.to_dict(),
            "child_predicate": self.inner.to_dict(),
        }
        legacy = _legacy_selector_fields(self.selector)
        if legacy:
            axis_str, child_type = legacy
            if child_type:
                result["child_type"] = child_type
            if axis_str != "child":
                result["child_axis"] = axis_str
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def __repr__(self) -> str:
        axis_prefix = "//" if self.selector.axis == Axis.DESC else ""
        return f"agg_{self.agg_type}({axis_prefix}{self.selector.test}[{self.inner}])"


@dataclass
class AggExistsPredicate(AggPredicate):
    """Compatibility wrapper for agg_exists."""
    def __init__(
        self,
        inner: PredicateNode,
        child_type: Optional[str] = None,
        child_axis: str = "child",
        selector: Optional[EvidenceSelector] = None,
        span: Optional[SourceSpan] = None,
    ):
        if selector is None:
            axis = Axis.from_str(child_axis)
            test = _selector_test_from_child_type(child_type)
            selector = EvidenceSelector(axis=axis, test=test)
        super().__init__(agg_type="exists", selector=selector, inner=inner, span=span)

    @property
    def child_type(self) -> Optional[str]:
        legacy = _legacy_selector_fields(self.selector)
        return legacy[1] if legacy else None

    @property
    def child_axis(self) -> str:
        legacy = _legacy_selector_fields(self.selector)
        return legacy[0] if legacy else "child"

    def __repr__(self) -> str:
        return AggPredicate.__repr__(self)


@dataclass
class AggPrevPredicate(AggPredicate):
    """Compatibility wrapper for agg_prev."""
    def __init__(
        self,
        inner: PredicateNode,
        child_type: Optional[str] = None,
        child_axis: str = "child",
        selector: Optional[EvidenceSelector] = None,
        span: Optional[SourceSpan] = None,
    ):
        if selector is None:
            axis = Axis.from_str(child_axis)
            test = _selector_test_from_child_type(child_type)
            selector = EvidenceSelector(axis=axis, test=test)
        super().__init__(agg_type="prev", selector=selector, inner=inner, span=span)

    @property
    def child_type(self) -> Optional[str]:
        legacy = _legacy_selector_fields(self.selector)
        return legacy[1] if legacy else None

    @property
    def child_axis(self) -> str:
        legacy = _legacy_selector_fields(self.selector)
        return legacy[0] if legacy else "child"

    def __repr__(self) -> str:
        return AggPredicate.__repr__(self)


# =============================================================================
# Logical combinators (recursive)
# =============================================================================

@dataclass
class AndPredicate(PredicateNode):
    """
    ψ₁ AND ψ₂ AND ... - conjunction.

    Paper: Score(u, ψ₁ ∧ ψ₂) = min{Score(u, ψ₁), Score(u, ψ₂)}
    """
    children: List[PredicateNode] = field(default_factory=list)
    span: Optional[SourceSpan] = None

    def get_all_atomic_values(self) -> List[str]:
        values: List[str] = []
        for child in self.children:
            values.extend(child.get_all_atomic_values())
        return values

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "operator": "AND",
            "conditions": [c.to_dict() for c in self.children],
        }
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def __repr__(self) -> str:
        return " AND ".join(str(c) for c in self.children)


@dataclass
class OrPredicate(PredicateNode):
    """
    ψ₁ OR ψ₂ OR ... - disjunction.

    Paper: Score(u, ψ₁ ∨ ψ₂) = max{Score(u, ψ₁), Score(u, ψ₂)}
    """
    children: List[PredicateNode] = field(default_factory=list)
    span: Optional[SourceSpan] = None

    def get_all_atomic_values(self) -> List[str]:
        values: List[str] = []
        for child in self.children:
            values.extend(child.get_all_atomic_values())
        return values

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "operator": "OR",
            "conditions": [c.to_dict() for c in self.children],
        }
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def __repr__(self) -> str:
        return " OR ".join(str(c) for c in self.children)


@dataclass
class NotPredicate(PredicateNode):
    """
    not(ψ) - negation.

    Paper: Score(u, ¬ψ) = 1 - Score(u, ψ)
    """
    child: PredicateNode = field(default=None)  # type: ignore[assignment]
    span: Optional[SourceSpan] = None

    def get_all_atomic_values(self) -> List[str]:
        return self.child.get_all_atomic_values() if self.child else []

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "operator": "NOT",
            "condition": self.child.to_dict() if self.child else {},
        }
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def __repr__(self) -> str:
        return f"not({self.child})"


def _selector_test_from_child_type(child_type: Optional[str]) -> NodeTestExpr:
    if child_type:
        test = NodeTest(kind="type", name=child_type)
    else:
        test = NodeTest(kind="wildcard", name=None)
    return NodeTestLeaf(test=test)


def _legacy_selector_fields(selector: EvidenceSelector) -> Optional[Tuple[str, Optional[str]]]:
    if isinstance(selector.test, NodeTestLeaf):
        test = selector.test.test
        if test.kind == "type":
            return (selector.axis.value if selector.axis != Axis.NONE else "child", test.name)
        if test.kind == "wildcard":
            return (selector.axis.value if selector.axis != Axis.NONE else "child", None)
    return None
