"""
Semantic XPath Parser - Parses queries into structured ASTs.

Pipeline:
  1) Global index extraction
  2) Path decomposition
  3) Step parsing (Axis + NodeTestExpr)
  4) Predicate parsing (boolean + aggregation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

from .parsing_models import (
    Axis,
    EvidenceSelector,
    Index,
    IndexRange,
    NodeTest,
    NodeTestAnd,
    NodeTestExpr,
    NodeTestLeaf,
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
    AggExistsPredicate,
    AggPrevPredicate,
    AndPredicate,
    OrPredicate,
    NotPredicate,
    Token,
    TokenType,
    tokenize,
)


# =============================================================================
# Errors
# =============================================================================


class QueryParseError(Exception):
    """Raised when query parsing fails."""


class NodeTestParseError(Exception):
    """Raised when node test expression parsing fails."""


class PredicateParseError(Exception):
    """Raised when predicate parsing fails."""


# =============================================================================
# Index Parsing
# =============================================================================


def parse_index(text: str, span: Optional[SourceSpan] = None) -> Optional[Index]:
    """Parse an index string like '2', '-1', '1:3', or '-2:'."""
    s = text.strip()
    if not s:
        return None

    if s.count(":") > 1:
        return None

    if ":" in s:
        start_str, end_str = s.split(":", 1)
        start_str = start_str.strip()
        end_str = end_str.strip()
        if start_str == "":
            return None
        try:
            start = int(start_str)
        except ValueError:
            return None
        if start == 0:
            raise NodeTestParseError("Index 0 is invalid (1-based indexing).")
        if end_str == "":
            return Index(start=start, end=None, to_end=True, span=span)
        try:
            end = int(end_str)
        except ValueError:
            return None
        if end == 0:
            raise NodeTestParseError("Index 0 is invalid (1-based indexing).")
        return Index(start=start, end=end, span=span)

    try:
        start = int(s)
    except ValueError:
        return None
    if start == 0:
        raise NodeTestParseError("Index 0 is invalid (1-based indexing).")
    return Index(start=start, span=span)


# =============================================================================
# Node Test Expression Parser
# =============================================================================


class NodeTestExprParser:
    """
    Recursive-descent parser for node test expressions.

    Grammar (AND binds tighter than OR):
        expr      := or_expr
        or_expr   := and_expr (OR and_expr)*
        and_expr  := primary (AND primary)*
        primary   := node_test | LPAREN expr RPAREN
        node_test := IDENT | "*"  with optional [index] and [predicate]
    """

    def __init__(
        self,
        text: str,
        base_offset: int = 0,
        predicate_parser: Optional[Callable[[str, int], PredicateNode]] = None,
    ):
        self._text = text
        self._pos = 0
        self._base_offset = base_offset
        self._predicate_parser = predicate_parser or parse_predicate

    def parse(self) -> NodeTestExpr:
        expr = self._parse_or()
        self._skip_ws()
        if self._pos != len(self._text):
            raise NodeTestParseError(
                f"Unexpected trailing input at pos {self._base_offset + self._pos}"
            )
        return expr

    def parse_with_pos(self) -> Tuple[NodeTestExpr, int]:
        expr = self._parse_or()
        self._skip_ws()
        return expr, self._pos

    def _skip_ws(self) -> None:
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> str:
        return self._text[self._pos] if self._pos < len(self._text) else ""

    def _match_keyword(self, kw: str) -> bool:
        self._skip_ws()
        end = self._pos + len(kw)
        if self._text[self._pos:end].upper() == kw.upper():
            next_char = self._text[end:end + 1]
            if next_char and (next_char.isalnum() or next_char == "_"):
                return False
            self._pos = end
            return True
        return False

    def _parse_or(self) -> NodeTestExpr:
        left = self._parse_and()
        parts = [left]
        while self._match_keyword("OR"):
            parts.append(self._parse_and())
        if len(parts) == 1:
            return parts[0]
        start = parts[0].span.start if parts[0].span else None
        end = parts[-1].span.end if parts[-1].span else None
        span = SourceSpan(start, end) if start is not None and end is not None else None
        return NodeTestOr(children=parts, span=span)

    def _parse_and(self) -> NodeTestExpr:
        left = self._parse_primary()
        parts = [left]
        while self._match_keyword("AND"):
            parts.append(self._parse_primary())
        if len(parts) == 1:
            return parts[0]
        start = parts[0].span.start if parts[0].span else None
        end = parts[-1].span.end if parts[-1].span else None
        span = SourceSpan(start, end) if start is not None and end is not None else None
        return NodeTestAnd(children=parts, span=span)

    def _parse_primary(self) -> NodeTestExpr:
        self._skip_ws()
        if self._peek() == "(":
            start = self._pos
            self._pos += 1
            expr = self._parse_or()
            self._skip_ws()
            if self._peek() != ")":
                raise NodeTestParseError(
                    f"Expected ')' at pos {self._base_offset + self._pos}"
                )
            self._pos += 1
            expr.span = SourceSpan(self._base_offset + start, self._base_offset + self._pos)
            return expr
        return self._parse_node_test()

    def _parse_node_test(self) -> NodeTestExpr:
        self._skip_ws()
        start = self._pos
        if self._peek() in ("*", "."):
            self._pos += 1
            kind = "wildcard"
            name = None
        else:
            name = self._parse_ident()
            if not name:
                raise NodeTestParseError(
                    f"Expected node test at pos {self._base_offset + self._pos}"
                )
            kind = "type"

        index: Optional[Index] = None
        predicate: Optional[PredicateNode] = None

        while True:
            self._skip_ws()
            if self._peek() != "[":
                break
            content, bracket_span = self._extract_bracket()
            idx = parse_index(content, span=bracket_span)
            if idx:
                if index is not None:
                    raise NodeTestParseError("Multiple indices on a node test.")
                index = idx
                continue
            pred = self._predicate_parser(content, bracket_span.start + 1)
            if pred is None:
                raise NodeTestParseError(
                    f"Invalid predicate in brackets at pos {bracket_span.start}"
                )
            if predicate is not None:
                raise NodeTestParseError("Multiple predicates on a node test.")
            predicate = pred

        end = self._pos
        test = NodeTest(
            kind=kind,
            name=name,
            index=index,
            predicate=predicate,
            span=SourceSpan(self._base_offset + start, self._base_offset + end),
        )
        return NodeTestLeaf(test=test, span=test.span)

    def _parse_ident(self) -> str:
        self._skip_ws()
        start = self._pos
        if start >= len(self._text):
            return ""
        if not (self._text[start].isalpha() or self._text[start] == "_"):
            return ""
        self._pos += 1
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch.isalnum() or ch == "_":
                self._pos += 1
            else:
                break
        return self._text[start:self._pos]

    def _extract_bracket(self) -> Tuple[str, SourceSpan]:
        if self._peek() != "[":
            raise NodeTestParseError(
                f"Expected '[' at pos {self._base_offset + self._pos}"
            )
        start = self._pos
        self._pos += 1
        depth = 1
        in_quote: Optional[str] = None
        content_start = self._pos
        while self._pos < len(self._text) and depth > 0:
            ch = self._text[self._pos]
            if in_quote:
                if ch == in_quote:
                    in_quote = None
            else:
                if ch in ('"', "'"):
                    in_quote = ch
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
            self._pos += 1
        if depth != 0:
            raise NodeTestParseError(
                f"Unterminated '[' starting at pos {self._base_offset + start}"
            )
        content_end = self._pos - 1
        content = self._text[content_start:content_end]
        span = SourceSpan(self._base_offset + start, self._base_offset + self._pos)
        return content, span


# =============================================================================
# Predicate Parser
# =============================================================================


class PredicateParser:
    """
    Recursive-descent parser for predicate expressions.

    Grammar (NOT > AND > OR):
        predicate   := or_expr
        or_expr     := and_expr ( OR and_expr )*
        and_expr    := unary_expr ( AND unary_expr )*
        unary_expr  := NOT unary_expr | primary
        primary     := atom_expr | agg_expr | LPAREN predicate RPAREN
        atom_expr   := ATOM LPAREN IDENT TILDE_EQ STRING RPAREN
                     | IDENT TILDE_EQ STRING
        agg_expr    := (AGG_EXISTS|AGG_PREV) LPAREN agg_inner RPAREN
    """

    def __init__(self, tokens: List[Token], source_text: str, base_offset: int = 0):
        self._tokens = tokens
        self._text = source_text
        self._pos = 0
        self._base_offset = base_offset

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._peek()
        if tok.type != tt:
            raise PredicateParseError(
                f"Expected {tt.name} but got {tok.type.name} ({tok.value!r}) at pos {tok.pos}"
            )
        return self._advance()

    def _match(self, tt: TokenType) -> Optional[Token]:
        if self._peek().type == tt:
            return self._advance()
        return None

    def parse(self) -> PredicateNode:
        node = self._parse_or()
        if self._peek().type != TokenType.EOF:
            tok = self._peek()
            raise PredicateParseError(
                f"Unexpected token {tok.type.name} ({tok.value!r}) at pos {tok.pos}"
            )
        return node

    def _parse_or(self) -> PredicateNode:
        left = self._parse_and()
        parts = [left]
        while self._match(TokenType.OR):
            parts.append(self._parse_and())
        if len(parts) == 1:
            return parts[0]
        span = SourceSpan(parts[0].span.start, parts[-1].span.end)
        return OrPredicate(children=parts, span=span)

    def _parse_and(self) -> PredicateNode:
        left = self._parse_unary()
        parts = [left]
        while self._match(TokenType.AND):
            parts.append(self._parse_unary())
        if len(parts) == 1:
            return parts[0]
        span = SourceSpan(parts[0].span.start, parts[-1].span.end)
        return AndPredicate(children=parts, span=span)

    def _parse_unary(self) -> PredicateNode:
        if self._peek().type == TokenType.NOT:
            tok = self._advance()
            child = self._parse_unary()
            span = SourceSpan(tok.pos, child.span.end)
            return NotPredicate(child=child, span=span)
        return self._parse_primary()

    def _parse_primary(self) -> PredicateNode:
        tt = self._peek().type

        if tt == TokenType.ATOM:
            return self._parse_atom_wrapped()

        if tt in (TokenType.AGG_EXISTS, TokenType.AGG_PREV):
            return self._parse_agg_expr(tt == TokenType.AGG_EXISTS)

        if tt == TokenType.LPAREN:
            lparen = self._advance()
            inner = self._parse_or()
            rparen = self._expect(TokenType.RPAREN)
            inner.span = SourceSpan(lparen.pos, rparen.end)
            return inner

        if tt == TokenType.IDENT:
            return self._parse_atom_direct()

        tok = self._peek()
        raise PredicateParseError(
            f"Expected atom/agg/( but got {tok.type.name} ({tok.value!r}) at pos {tok.pos}"
        )

    def _parse_atom_wrapped(self) -> AtomPredicate:
        tok = self._expect(TokenType.ATOM)
        self._expect(TokenType.LPAREN)
        field_tok = self._expect(TokenType.IDENT)
        self._expect(TokenType.TILDE_EQ)
        value_tok = self._expect(TokenType.STRING)
        rparen = self._expect(TokenType.RPAREN)
        span = SourceSpan(tok.pos, rparen.end)
        return AtomPredicate(
            field=field_tok.value,
            operator="=~",
            value=value_tok.value,
            span=span,
        )

    def _parse_atom_direct(self) -> AtomPredicate:
        field_tok = self._expect(TokenType.IDENT)
        if self._peek().type != TokenType.TILDE_EQ:
            raise PredicateParseError(
                f"Expected '=~' after field at pos {self._peek().pos}"
            )
        self._advance()
        value_tok = self._expect(TokenType.STRING)
        span = SourceSpan(field_tok.pos, value_tok.end)
        return AtomPredicate(
            field=field_tok.value,
            operator="=~",
            value=value_tok.value,
            span=span,
        )

    def _parse_agg_expr(self, is_exists: bool) -> PredicateNode:
        agg_tok = self._advance()  # consume agg keyword
        lparen = self._expect(TokenType.LPAREN)

        # Find matching RPAREN in token stream
        depth = 1
        idx = self._pos
        while idx < len(self._tokens):
            tok = self._tokens[idx]
            if tok.type == TokenType.LPAREN:
                depth += 1
            elif tok.type == TokenType.RPAREN:
                depth -= 1
                if depth == 0:
                    break
            idx += 1
        if idx >= len(self._tokens) or depth != 0:
            raise PredicateParseError(
                f"Unterminated agg expression starting at pos {agg_tok.pos}"
            )

        inner_end_tok = self._tokens[idx]
        inner_start = lparen.end - self._base_offset
        inner_end = inner_end_tok.pos - self._base_offset
        inner_text = self._text[inner_start:inner_end]
        selector, inner_pred = _parse_agg_inner(
            inner_text,
            base_offset=lparen.end,
        )

        # Advance parser position beyond the RPAREN
        self._pos = idx + 1

        span = SourceSpan(agg_tok.pos, inner_end_tok.end)
        if is_exists:
            return AggExistsPredicate(inner=inner_pred, selector=selector, span=span)
        return AggPrevPredicate(inner=inner_pred, selector=selector, span=span)


def parse_predicate(text: str, offset: int = 0) -> PredicateNode:
    """Convenience: tokenize + parse a predicate string."""
    tokens = tokenize(text, offset=offset)
    return PredicateParser(tokens, text, base_offset=offset).parse()


# =============================================================================
# Aggregate Inner Parsing Helpers
# =============================================================================


def _parse_agg_inner(text: str, base_offset: int) -> Tuple[EvidenceSelector, PredicateNode]:
    """
    Parse aggregate inner content.

    Supported forms:
      - [axis::]NodeTestExpr[inner_pred]
      - predicate_expr     (no selector)

    If selector has a single leaf predicate and no explicit inner,
    move that predicate to the inner predicate for compatibility.
    """
    stripped = text.strip()
    if stripped == "":
        raise PredicateParseError("Empty aggregate predicate body.")

    selector = _try_parse_selector(text, base_offset)
    if selector:
        selector_obj, _remaining, inner_bracket = selector
        if inner_bracket is not None:
            inner_pred = parse_predicate(inner_bracket.value, offset=inner_bracket._offset)
            return selector_obj, inner_pred

        # No explicit inner: lift predicate from single leaf if present
        leaf = _single_leaf(selector_obj.test)
        if leaf and leaf.test.predicate:
            inner_pred = leaf.test.predicate
            leaf.test.predicate = None
            return selector_obj, inner_pred

        raise PredicateParseError("Aggregate predicate requires an inner predicate.")

    # Fallback: parse as predicate-only (selector wildcard)
    inner_pred = parse_predicate(stripped, offset=base_offset + (len(text) - len(text.lstrip())))
    selector_obj = EvidenceSelector(axis=Axis.NONE, test=NodeTestLeaf(NodeTest(kind="wildcard")))
    return selector_obj, inner_pred


@dataclass
class _BracketPayload:
    value: str
    _offset: int


def _try_parse_selector(
    text: str, base_offset: int
) -> Optional[Tuple[EvidenceSelector, str, Optional[_BracketPayload]]]:
    """
    Try to parse a selector from the start of the text.

    Supports multi-step path selectors like Breakfast/Meal[pred].
    Intermediate steps are stored as path_prefix on the EvidenceSelector.

    Returns (selector, remaining_text, optional_inner_bracket).
    """
    pos = 0
    while pos < len(text) and text[pos].isspace():
        pos += 1
    offset = base_offset + pos
    remaining = text[pos:]

    axis, axis_len = _parse_axis_prefix(remaining)
    remaining = remaining[axis_len:]
    offset += axis_len

    try:
        parser = NodeTestExprParser(remaining, base_offset=offset, predicate_parser=parse_predicate)
        expr, consumed = parser.parse_with_pos()
    except NodeTestParseError:
        return None

    rest = remaining[consumed:]
    current_offset = offset + consumed

    # Handle multi-step paths: Breakfast/Meal[pred]
    # Collect intermediate steps as path_prefix
    path_prefix = []
    while rest.startswith("/") and not rest.startswith("//"):
        # The previous expr becomes a path prefix step
        if not isinstance(expr, NodeTestLeaf) or expr.test.kind != "type":
            break  # Only simple type names can be path prefix steps
        if expr.test.predicate or expr.test.index:
            break  # Prefix steps shouldn't have predicates or indices
        path_prefix.append(expr.test.name)

        # Advance past the "/"
        rest = rest[1:]
        current_offset += 1

        # Parse the next step
        try:
            parser = NodeTestExprParser(rest, base_offset=current_offset, predicate_parser=parse_predicate)
            expr, consumed = parser.parse_with_pos()
        except NodeTestParseError:
            return None

        rest = rest[consumed:]
        current_offset += consumed

    # Skip whitespace
    rest_pos = 0
    while rest_pos < len(rest) and rest[rest_pos].isspace():
        rest_pos += 1
    rest = rest[rest_pos:]

    inner_bracket: Optional[_BracketPayload] = None
    if rest:
        if rest[0] != "[":
            return None
        content, end_pos = _extract_bracket_from_string(rest, current_offset + rest_pos)
        if rest[end_pos:].strip():
            return None
        inner_bracket = _BracketPayload(value=content, _offset=current_offset + rest_pos + 1)

    selector = EvidenceSelector(
        axis=axis,
        test=expr,
        path_prefix=path_prefix,
        span=SourceSpan(offset, current_offset)
    )
    return selector, rest, inner_bracket



def _extract_bracket_from_string(text: str, base_offset: int) -> Tuple[str, int]:
    if not text.startswith("["):
        raise PredicateParseError("Expected '[' in aggregate selector.")
    depth = 1
    in_quote: Optional[str] = None
    i = 1
    while i < len(text) and depth > 0:
        ch = text[i]
        if in_quote:
            if ch == in_quote:
                in_quote = None
        else:
            if ch in ('"', "'"):
                in_quote = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
        i += 1
    if depth != 0:
        raise PredicateParseError(f"Unterminated '[' at pos {base_offset}")
    return text[1:i - 1], i


def _single_leaf(expr: NodeTestExpr) -> Optional[NodeTestLeaf]:
    return expr if isinstance(expr, NodeTestLeaf) else None


# =============================================================================
# Query Parser
# =============================================================================


class QueryParser:
    """
    Parses Semantic XPath queries into structured ASTs.
    """

    def parse(self, query: str) -> Query:
        if not query:
            raise QueryParseError("Empty query.")

        start = _skip_leading_ws(query, 0)
        end = _skip_trailing_ws(query, len(query))
        text = query[start:end]

        inner_text = text
        inner_offset = start
        global_index: Optional[Index] = None

        if text.startswith("("):
            close_idx = _find_matching_paren(text, 0)
            if close_idx is not None:
                after = text[close_idx + 1:].lstrip()
                if after.startswith("[") and text.rstrip().endswith("]"):
                    bracket_start = close_idx + 1 + (len(text[close_idx + 1:]) - len(after))
                    idx_content, bracket_end = _extract_bracket_global(text, bracket_start)
                    if text[bracket_end:].strip() == "":
                        idx_span = SourceSpan(
                            inner_offset + bracket_start,
                            inner_offset + bracket_end,
                        )
                        idx = parse_index(idx_content, span=idx_span)
                        if idx is None:
                            raise QueryParseError("Invalid global index.")
                        global_index = idx
                        inner_text = text[1:close_idx]
                        inner_offset = start + 1

        # Check for union operator (|) - split and parse each path, then combine
        union_paths = self._split_union(inner_text, inner_offset)
        if len(union_paths) > 1:
            # Parse each path separately and combine into a union path
            parsed_paths = [self._parse_path(path_text, path_offset) for path_text, path_offset in union_paths]
            # Create a synthetic path that represents the union by combining all steps
            # For now, we'll use the first path but mark it as a union
            # TODO: Update PathExpr/Query to properly support unions
            path = parsed_paths[0]  # Use first path for now - executor will need updates for full union support
        else:
            path = self._parse_path(inner_text, inner_offset)
        
        span = SourceSpan(start, end)
        return Query(path=path, global_index=global_index, span=span)

    def parse_legacy(self, query: str) -> Tuple[List[QueryStep], Optional[IndexRange]]:
        """
        Legacy parse mode returning QueryStep + IndexRange for backward compatibility.
        """
        parsed = self.parse(query)
        steps: List[QueryStep] = []
        for step in parsed.path.steps:
            if not isinstance(step.test, NodeTestLeaf):
                raise QueryParseError("Legacy parse supports only simple node tests.")
            test = step.test.test
            if test.kind == "wildcard":
                node_type = "*"
            else:
                node_type = test.name or "?"
            axis = step.axis.value if step.axis != Axis.NONE else "child"
            index = None
            if test.index:
                index = IndexRange(
                    start=test.index.start,
                    end=test.index.end,
                    to_end=test.index.to_end,
                )
            steps.append(QueryStep(
                node_type=node_type,
                predicate=test.predicate,
                index=index,
                axis=axis,
            ))

        global_index = None
        if parsed.global_index:
            global_index = IndexRange(
                start=parsed.global_index.start,
                end=parsed.global_index.end,
                to_end=parsed.global_index.to_end,
            )
        return steps, global_index

    def _split_union(self, text: str, base_offset: int) -> List[Tuple[str, int]]:
        """
        Split a query on union operators (| or OR) while respecting brackets, parens, and quotes.
        
        Returns:
            List of (path_text, path_offset) tuples for each union path
        """
        parts: List[Tuple[str, int]] = []
        bracket_depth = 0
        paren_depth = 0
        in_quote: Optional[str] = None
        current_start = 0
        
        i = 0
        while i < len(text):
            ch = text[i]
            if in_quote:
                if ch == in_quote:
                    in_quote = None
                i += 1
                continue
            if ch in ('"', "'"):
                in_quote = ch
                i += 1
                continue
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "|" and bracket_depth == 0 and paren_depth == 0:
                # Found a pipe union operator at top level
                segment = text[current_start:i].strip()
                if segment:
                    parts.append((segment, base_offset + current_start))
                current_start = i + 1
            elif bracket_depth == 0 and paren_depth == 0:
                # Check for keyword OR at top level (case-insensitive)
                remaining = text[i:]
                if remaining[:2].upper() == "OR" and len(remaining) > 2 and not remaining[2].isalnum() and remaining[2] != "_":
                    # Verify it's preceded by whitespace or start of string
                    if i == 0 or text[i - 1].isspace():
                        segment = text[current_start:i].strip()
                        if segment:
                            parts.append((segment, base_offset + current_start))
                        current_start = i + 2  # skip past "OR"
                        i += 2
                        continue
            i += 1
        
        # Add final segment
        if current_start < len(text):
            segment = text[current_start:].strip()
            if segment:
                parts.append((segment, base_offset + current_start))
        
        # If no union found, return single path
        if not parts:
            return [(text, base_offset)]
        
        return parts

    def _parse_path(self, text: str, base_offset: int) -> PathExpr:
        # Handle parenthesized path expressions: (path)[predicate][index]
        # Strip leading/trailing whitespace for detection
        trimmed = text.strip()
        trim_start = len(text) - len(trimmed.lstrip())
        
        if trimmed.startswith("("):
            close_idx = _find_matching_paren(trimmed, 0)
            if close_idx is not None:
                # Extract the inner path
                inner_path = trimmed[1:close_idx].strip()
                after_paren = trimmed[close_idx + 1:].strip()
                
                # Parse the inner path
                inner_offset = base_offset + trim_start + 1
                path_expr = self._parse_path(inner_path, inner_offset)
                
                # If there are predicates/indexes after the closing paren,
                # extract them and apply to the last step
                if after_paren.startswith("["):
                    # Extract all bracket pairs after the closing paren
                    bracket_base = base_offset + trim_start + close_idx + 1
                    remaining = after_paren
                    last_step = path_expr.steps[-1] if path_expr.steps else None
                    
                    if last_step and isinstance(last_step.test, NodeTestLeaf):
                        # Extract predicates and indexes from brackets
                        pos_in_after = 0  # Position within after_paren string
                        while pos_in_after < len(after_paren) and after_paren[pos_in_after:].strip().startswith("["):
                            # Skip whitespace
                            while pos_in_after < len(after_paren) and after_paren[pos_in_after].isspace():
                                pos_in_after += 1
                            
                            if pos_in_after >= len(after_paren) or after_paren[pos_in_after] != "[":
                                break
                            
                            bracket_content, bracket_len = _extract_bracket_from_string(
                                after_paren[pos_in_after:], 
                                bracket_base + pos_in_after
                            )
                            bracket_start_pos = bracket_base + pos_in_after + 1  # +1 to skip '['
                            
                            # Try to parse as index first
                            idx_span = SourceSpan(bracket_start_pos, bracket_base + pos_in_after + bracket_len - 1)
                            idx = parse_index(bracket_content, span=idx_span)
                            if idx:
                                # Apply index to last step
                                if last_step.test.test.index is not None:
                                    raise QueryParseError("Multiple indices on a node test.")
                                last_step.test.test.index = idx
                            else:
                                # Parse as predicate
                                pred = parse_predicate(bracket_content, bracket_start_pos)
                                if pred:
                                    # Combine with existing predicate if any
                                    if last_step.test.test.predicate:
                                        # Combine predicates with AND
                                        from .predicate_ast import AndPredicate
                                        combined = AndPredicate(
                                            children=[last_step.test.test.predicate, pred],
                                            span=SourceSpan(
                                                last_step.test.test.predicate.span.start,
                                                pred.span.end
                                            )
                                        )
                                        last_step.test.test.predicate = combined
                                    else:
                                        last_step.test.test.predicate = pred
                            
                            # Advance past this bracket
                            pos_in_after += bracket_len
                            # Skip whitespace
                            while pos_in_after < len(after_paren) and after_paren[pos_in_after].isspace():
                                pos_in_after += 1
                    
                    # Update span to include the brackets
                    if path_expr.steps:
                        final_end = bracket_base + len(after_paren)
                        path_expr.span = SourceSpan(
                            path_expr.span.start,
                            final_end
                        )
                
                return path_expr
        
        steps: List[Step] = []
        for part, start, end, axis_hint in _split_path(text, base_offset):
            step = self._parse_step(part, start, end, axis_hint)
            steps.append(step)
        span = SourceSpan(base_offset, base_offset + len(text))
        return PathExpr(steps=steps, span=span)

    def _parse_step(self, text: str, start: int, end: int, axis_hint: Axis) -> Step:
        trimmed, trim_offset = _lstrip_with_offset(text)
        axis, axis_len = _parse_axis_prefix(trimmed)
        body = trimmed[axis_len:]
        if axis_len == 0:
            axis = axis_hint
        body_offset = start + trim_offset + axis_len
        parser = NodeTestExprParser(body, base_offset=body_offset, predicate_parser=parse_predicate)
        expr = parser.parse()
        span = SourceSpan(start, end)
        return Step(axis=axis, test=expr, span=span)


# =============================================================================
# Helpers
# =============================================================================


def _parse_axis_prefix(text: str) -> Tuple[Axis, int]:
    if text.startswith("//"):
        return Axis.DESC, len("//")
    if text.startswith("/"):
        return Axis.NONE, len("/")
    return Axis.NONE, 0


def _skip_leading_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _skip_trailing_ws(text: str, pos: int) -> int:
    while pos > 0 and text[pos - 1].isspace():
        pos -= 1
    return pos


def _lstrip_with_offset(text: str) -> Tuple[str, int]:
    original_len = len(text)
    stripped = text.lstrip()
    return stripped, original_len - len(stripped)


def _find_matching_paren(text: str, start: int) -> Optional[int]:
    if text[start] != "(":
        return None
    depth = 0
    in_quote: Optional[str] = None
    for i in range(start, len(text)):
        ch = text[i]
        if in_quote:
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ('"', "'"):
            in_quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _extract_bracket_global(text: str, start: int) -> Tuple[str, int]:
    if text[start] != "[":
        raise QueryParseError("Expected '[' for global index.")
    depth = 1
    in_quote: Optional[str] = None
    i = start + 1
    while i < len(text) and depth > 0:
        ch = text[i]
        if in_quote:
            if ch == in_quote:
                in_quote = None
        else:
            if ch in ('"', "'"):
                in_quote = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
        i += 1
    if depth != 0:
        raise QueryParseError("Unterminated global index bracket.")
    return text[start + 1:i - 1], i


def _split_path(text: str, base_offset: int) -> List[Tuple[str, int, int, Axis]]:
    parts: List[Tuple[str, int, int, Axis]] = []
    bracket_depth = 0
    paren_depth = 0
    in_quote: Optional[str] = None
    current_start = 0
    pending_axis = Axis.NONE

    i = 0
    while i < len(text):
        ch = text[i]
        if in_quote:
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_quote = ch
            i += 1
            continue
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "/" and bracket_depth == 0 and paren_depth == 0:
            is_double = i + 1 < len(text) and text[i + 1] == "/"
            segment = text[current_start:i]
            if segment:
                start = base_offset + current_start
                end = base_offset + i
                parts.append((segment, start, end, pending_axis))
                pending_axis = Axis.NONE
            if is_double:
                pending_axis = Axis.DESC
                i += 1
            else:
                pending_axis = Axis.NONE
            current_start = i + 1
        i += 1

    if current_start < len(text):
        segment = text[current_start:]
        if segment:
            start = base_offset + current_start
            end = base_offset + len(text)
            parts.append((segment, start, end, pending_axis))

    # Trim leading empty segment for absolute paths
    if parts and parts[0][0] == "":
        parts = parts[1:]
    return parts


# Singleton instance
_parser_instance: Optional[QueryParser] = None


def get_parser() -> QueryParser:
    """Get singleton parser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = QueryParser()
    return _parser_instance
