"""
Parsing Models - Data classes produced by the query parser.

Contains QueryStep and IndexRange, which are the output of parsing
a Semantic XPath query string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .predicate_ast import PredicateNode


# =============================================================================
# Source Spans
# =============================================================================

@dataclass(frozen=True)
class SourceSpan:
    """Inclusive-exclusive source span in the original query string."""
    start: int
    end: int

    def to_dict(self) -> Dict[str, int]:
        return {"start": self.start, "end": self.end}


# =============================================================================
# Axis and Index Models
# =============================================================================

class Axis(Enum):
    """Traversal axis for steps and evidence selectors."""
    NONE = "none"
    CHILD = "child"
    DESC = "desc"

    @classmethod
    def from_str(cls, value: Optional[str]) -> "Axis":
        if not value:
            return cls.NONE
        value = value.lower()
        if value == "child":
            return cls.CHILD
        if value == "desc":
            return cls.DESC
        return cls.NONE


@dataclass
class Index:
    """Represents an index or index range for positional selection."""
    start: int  # 1-based start index (or single index, or negative for from-end)
    end: Optional[int] = None  # 1-based end index (inclusive)
    to_end: bool = False  # If True, range extends to the end (legacy support)
    span: Optional[SourceSpan] = None

    @property
    def is_range(self) -> bool:
        return self.end is not None or self.to_end

    def __repr__(self) -> str:
        if self.to_end:
            return f"[{self.start}:]"
        if self.end is not None:
            return f"[{self.start}:{self.end}]"
        return f"[{self.start}]"

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"start": self.start}
        if self.to_end:
            result["type"] = "range_to_end"
            result["to_end"] = True
        elif self.end is not None:
            result["type"] = "range"
            result["end"] = self.end
        else:
            result["type"] = "single"
        if self.span:
            result["span"] = self.span.to_dict()
        return result


# =============================================================================
# Node Test Expression AST
# =============================================================================

@dataclass
class NodeTest:
    """Leaf node test."""
    kind: str  # "type" or "wildcard"
    name: Optional[str] = None
    index: Optional[Index] = None
    predicate: Optional["PredicateNode"] = None
    span: Optional[SourceSpan] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "node_test",
            "kind": self.kind,
            "name": self.name,
        }
        if self.index:
            result["index"] = self.index.to_dict()
        if self.predicate:
            result["predicate"] = self.predicate.to_dict()
            result["predicate_str"] = str(self.predicate)
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def __repr__(self) -> str:
        base = "*" if self.kind == "wildcard" else (self.name or "?")
        brackets: List[tuple[int, str]] = []
        if self.predicate:
            pred_span = getattr(self.predicate, "span", None)
            order = pred_span.start if pred_span else 0
            brackets.append((order, f"[{self.predicate}]"))
        if self.index:
            idx_span = self.index.span
            order = idx_span.start if idx_span else 1
            brackets.append((order, repr(self.index)))
        for _, frag in sorted(brackets, key=lambda item: item[0]):
            base += frag
        return base


class NodeTestExpr:
    """Base class for node test expressions."""
    span: Optional[SourceSpan]

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_all_predicates(self) -> Iterable["PredicateNode"]:
        return []


@dataclass
class NodeTestLeaf(NodeTestExpr):
    test: NodeTest
    span: Optional[SourceSpan] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": "leaf", "test": self.test.to_dict()}
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def get_all_predicates(self) -> Iterable["PredicateNode"]:
        return [self.test.predicate] if self.test.predicate else []

    def __repr__(self) -> str:
        return repr(self.test)


@dataclass
class NodeTestAnd(NodeTestExpr):
    children: List[NodeTestExpr] = field(default_factory=list)
    span: Optional[SourceSpan] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": "and",
            "children": [c.to_dict() for c in self.children],
        }
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def get_all_predicates(self) -> Iterable["PredicateNode"]:
        for child in self.children:
            yield from child.get_all_predicates()

    def __repr__(self) -> str:
        return " AND ".join(repr(c) for c in self.children)


@dataclass
class NodeTestOr(NodeTestExpr):
    children: List[NodeTestExpr] = field(default_factory=list)
    span: Optional[SourceSpan] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": "or",
            "children": [c.to_dict() for c in self.children],
        }
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def get_all_predicates(self) -> Iterable["PredicateNode"]:
        for child in self.children:
            yield from child.get_all_predicates()

    def __repr__(self) -> str:
        return " OR ".join(repr(c) for c in self.children)


# =============================================================================
# Evidence Selector and Query AST
# =============================================================================

@dataclass
class EvidenceSelector:
    axis: Axis
    test: NodeTestExpr
    path_prefix: List[str] = field(default_factory=list)
    span: Optional[SourceSpan] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "axis": self.axis.value,
            "test": self.test.to_dict(),
        }
        if self.path_prefix:
            result["path_prefix"] = self.path_prefix
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def __repr__(self) -> str:
        axis_prefix = "//" if self.axis == Axis.DESC else ""
        prefix_str = "/".join(self.path_prefix) + "/" if self.path_prefix else ""
        return f"{axis_prefix}{prefix_str}{self.test}"


@dataclass
class Step:
    axis: Axis
    test: NodeTestExpr
    span: Optional[SourceSpan] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "axis": self.axis.value,
            "test": self.test.to_dict(),
        }
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def __repr__(self) -> str:
        axis = f"{self.axis.value}::" if self.axis != Axis.NONE else ""
        return f"{axis}{self.test}"


@dataclass
class PathExpr:
    steps: List[Step] = field(default_factory=list)
    span: Optional[SourceSpan] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.span:
            result["span"] = self.span.to_dict()
        return result

    def __repr__(self) -> str:
        if not self.steps:
            return "/"
        parts: List[str] = []
        for step in self.steps:
            sep = "//" if step.axis == Axis.DESC else "/"
            parts.append(f"{sep}{step.test}")
        return "".join(parts)


@dataclass
class Query:
    path: PathExpr
    global_index: Optional[Index] = None
    span: Optional[SourceSpan] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"path": self.path.to_dict()}
        if self.global_index:
            result["global_index"] = self.global_index.to_dict()
        if self.span:
            result["span"] = self.span.to_dict()
        return result


# =============================================================================
# Index and Position Models
# =============================================================================

@dataclass
class IndexRange:
    """Represents an index or index range for positional selection."""
    start: int  # 1-based start index (or single index, or negative for from-end)
    end: Optional[int] = None  # 1-based end index (inclusive)
    to_end: bool = False  # If True, range extends to the end (for [-N:] syntax)

    @property
    def is_range(self) -> bool:
        """Check if this represents a range (vs single index)."""
        return self.end is not None or self.to_end

    def __repr__(self) -> str:
        if self.to_end:
            return f"[{self.start}:]"
        elif self.end is not None:
            return f"[{self.start}:{self.end}]"
        return f"[{self.start}]"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for tracing."""
        result: Dict[str, Any] = {"start": self.start}
        if self.to_end:
            result["type"] = "range_to_end"
            result["to_end"] = True
        elif self.end is not None:
            result["type"] = "range"
            result["end"] = self.end
        else:
            result["type"] = "single"
        return result


# =============================================================================
# Query Step
# =============================================================================

@dataclass
class QueryStep:
    """
    Represents a single step in an XPath query.

    Each step consists of:
    - axis: The traversal axis ("child" for direct children, "desc" for all descendants)
    - node_type: The target node type to match
    - predicate: Optional predicate AST for semantic filtering
    - index: Optional positional index or range
    """
    node_type: str
    predicate: Optional[PredicateNode] = None
    index: Optional[IndexRange] = None
    axis: str = "child"
    predicate_str: Optional[str] = None  # Original predicate string for display

    def __repr__(self) -> str:
        axis_prefix = "//" if self.axis == "desc" else ""
        parts = [f"{axis_prefix}{self.node_type}"]
        if self.predicate:
            parts.append(f'[{self.predicate}]')
        elif self.predicate_str:
            parts.append(f'[description =~ "{self.predicate_str}"]')
        if self.index is not None:
            parts.append(str(self.index))
        return "".join(parts)

    def has_semantic_predicate(self) -> bool:
        """Check if this step has a semantic predicate."""
        return self.predicate is not None
