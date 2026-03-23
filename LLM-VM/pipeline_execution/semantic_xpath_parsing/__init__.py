"""
Semantic XPath Parsing - Tokenizer, AST, and parser for Semantic XPath queries.
"""

from .parser import (
    QueryParser,
    get_parser,
    parse_predicate,
    PredicateParseError,
    QueryParseError,
    NodeTestParseError,
)
from .parsing_models import (
    Axis,
    EvidenceSelector,
    Index,
    IndexRange,
    NodeTest,
    NodeTestExpr,
    NodeTestLeaf,
    NodeTestAnd,
    NodeTestOr,
    PathExpr,
    Query,
    QueryStep,
    SourceSpan,
    Step,
)

from .predicate_ast import (
    PredicateNode,
    AtomPredicate,
    AggPredicate,
    AggExistsPredicate,
    AggPrevPredicate,
    AndPredicate,
    OrPredicate,
    NotPredicate,
    Token,
    TokenType,
    tokenize,
    TokenizeError,
)

__all__ = [
    # Parser
    "QueryParser",
    "get_parser",
    "parse_predicate",
    "PredicateParseError",
    "QueryParseError",
    "NodeTestParseError",
    # Parsing models
    "Axis",
    "EvidenceSelector",
    "Index",
    "IndexRange",
    "NodeTest",
    "NodeTestExpr",
    "NodeTestLeaf",
    "NodeTestAnd",
    "NodeTestOr",
    "PathExpr",
    "Query",
    "QueryStep",
    "SourceSpan",
    "Step",
    # AST nodes
    "PredicateNode",
    "AtomPredicate",
    "AggPredicate",
    "AggExistsPredicate",
    "AggPrevPredicate",
    "AndPredicate",
    "OrPredicate",
    "NotPredicate",
    # Tokenizer
    "Token",
    "TokenType",
    "tokenize",
    "TokenizeError",
]
