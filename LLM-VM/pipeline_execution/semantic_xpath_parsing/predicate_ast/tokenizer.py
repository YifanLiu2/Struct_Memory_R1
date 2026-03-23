"""
Predicate Tokenizer - Converts a predicate string into a flat token stream.

Token types cover the full predicate grammar:
    Keywords:    ATOM, AGG_EXISTS, AGG_PREV, NOT, AND, OR
    Symbols:     LPAREN, RPAREN, LBRACK, RBRACK, TILDE_EQ (=~), COLONCOLON (::), SLASH (/)
    Literals:    STRING ("museum"), IDENT (content, POI, desc, child)
    Control:     EOF
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class TokenType(Enum):
    """All token types understood by the predicate parser."""
    # Keywords
    ATOM = auto()
    AGG_EXISTS = auto()
    AGG_PREV = auto()
    NOT = auto()
    AND = auto()
    OR = auto()

    # Punctuation / operators
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACK = auto()      # [
    RBRACK = auto()      # ]
    TILDE_EQ = auto()    # =~
    COLONCOLON = auto()  # ::
    DOT = auto()         # . (self node / wildcard)
    STAR = auto()        # * (any node type)
    SLASH = auto()       # / (axis separator in agg selectors)

    # Literals
    STRING = auto()      # "museum" or 'museum'
    IDENT = auto()       # content, POI, desc, child, ...

    # End of input
    EOF = auto()


# Keywords recognised by the tokenizer.  Order matters: longer keywords first
# so that "agg_exists" is matched before "agg".
_KEYWORDS = {
    "agg_exists": TokenType.AGG_EXISTS,
    "agg_prev":   TokenType.AGG_PREV,
    "agg_or":     TokenType.AGG_EXISTS,
    "atom":       TokenType.ATOM,
    "not":        TokenType.NOT,
    "AND":        TokenType.AND,
    "OR":         TokenType.OR,
    "and":        TokenType.AND,
    "or":         TokenType.OR,
}


@dataclass
class Token:
    """A single token produced by the tokenizer."""
    type: TokenType
    value: str
    pos: int           # character offset in the source string
    end: int           # exclusive end offset in the source string

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, pos={self.pos}, end={self.end})"


class TokenizeError(Exception):
    """Raised when the tokenizer encounters unexpected input."""
    def __init__(self, message: str, pos: int):
        super().__init__(f"Tokenize error at pos {pos}: {message}")
        self.pos = pos


def tokenize(text: str, offset: int = 0) -> List[Token]:
    """
    Convert a predicate string into a list of tokens.

    Example::

        tokenize('atom(content =~ "museum") OR atom(content =~ "art")')
        # => [Token(ATOM, 'atom', 0), Token(LPAREN, '(', 4), ...]
    """
    tokens: List[Token] = []
    pos = 0
    length = len(text)

    while pos < length:
        # Skip whitespace
        if text[pos].isspace():
            pos += 1
            continue

        # Two-character operators
        if text[pos:pos+2] == "=~":
            tokens.append(Token(TokenType.TILDE_EQ, "=~", pos + offset, pos + offset + 2))
            pos += 2
            continue

        if text[pos:pos+2] == "::":
            tokens.append(Token(TokenType.COLONCOLON, "::", pos + offset, pos + offset + 2))
            pos += 2
            continue

        # Single-character punctuation
        if text[pos] == '(':
            tokens.append(Token(TokenType.LPAREN, "(", pos + offset, pos + offset + 1))
            pos += 1
            continue
        if text[pos] == ')':
            tokens.append(Token(TokenType.RPAREN, ")", pos + offset, pos + offset + 1))
            pos += 1
            continue
        if text[pos] == '[':
            tokens.append(Token(TokenType.LBRACK, "[", pos + offset, pos + offset + 1))
            pos += 1
            continue
        if text[pos] == ']':
            tokens.append(Token(TokenType.RBRACK, "]", pos + offset, pos + offset + 1))
            pos += 1
            continue
        if text[pos] == '.':
            tokens.append(Token(TokenType.DOT, ".", pos + offset, pos + offset + 1))
            pos += 1
            continue
        if text[pos] == '*':
            tokens.append(Token(TokenType.STAR, "*", pos + offset, pos + offset + 1))
            pos += 1
            continue
        if text[pos] == '/':
            tokens.append(Token(TokenType.SLASH, "/", pos + offset, pos + offset + 1))
            pos += 1
            continue

        # Quoted strings
        if text[pos] in ('"', "'"):
            quote = text[pos]
            start = pos
            pos += 1
            while pos < length and text[pos] != quote:
                pos += 1
            if pos >= length:
                raise TokenizeError(f"Unterminated string starting with {quote}", start)
            value = text[start+1:pos]
            tokens.append(Token(TokenType.STRING, value, start + offset, pos + offset + 1))
            pos += 1  # skip closing quote
            continue

        # Keywords and identifiers (must start with letter or _)
        if text[pos].isalpha() or text[pos] == '_':
            start = pos
            while pos < length and (text[pos].isalnum() or text[pos] == '_'):
                pos += 1
            word = text[start:pos]

            # Check keywords
            if word.lower() in _KEYWORDS:
                tokens.append(Token(_KEYWORDS[word.lower()], word, start + offset, pos + offset))
            else:
                tokens.append(Token(TokenType.IDENT, word, start + offset, pos + offset))
            continue

        raise TokenizeError(f"Unexpected character: {text[pos]!r}", pos)

    tokens.append(Token(TokenType.EOF, "", pos + offset, pos + offset))
    return tokens
