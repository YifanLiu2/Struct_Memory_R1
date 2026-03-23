"""
Predicate AST package - typed node hierarchy and tokenizer for semantic XPath predicates.
"""

from .nodes import (
    PredicateNode,
    AtomPredicate,
    AggPredicate,
    AggExistsPredicate,
    AggPrevPredicate,
    AndPredicate,
    OrPredicate,
    NotPredicate,
)
from .tokenizer import Token, TokenType, tokenize, TokenizeError

__all__ = [
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
