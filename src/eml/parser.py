"""Parsers for the textual and RPN forms of EML trees.

Textual form (accepted by :func:`parse`)::

    eml(1, eml(x, 1))

Reverse-Polish form (accepted by :func:`parse_rpn`)::

    x 1 eml 1 eml

Variable names are identifiers (letters, digits, ``_``, not starting with a digit).
The literal ``1`` is the only numeric atom. Any other bare number is a syntax
error; numbers like ``2`` must be written as EML trees over 1.
"""

from __future__ import annotations

import re
from typing import Iterator, List, Tuple

from eml.ast import Eml, Node, One, Var

_TOKEN_RE = re.compile(r"\s*(?:(eml)|(1)|([A-Za-z_][A-Za-z0-9_]*)|([(),]))")


class ParseError(ValueError):
    """Raised when EML source text cannot be parsed."""


def _tokenize(src: str) -> List[Tuple[str, str]]:
    """Return a list of (kind, text) tokens. Kind is one of
    {'eml', 'one', 'var', 'lparen', 'rparen', 'comma'}.
    """
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise ParseError(f"Unexpected character at position {pos}: {src[pos]!r}")
        pos = m.end()
        if m.group(1):
            tokens.append(("eml", m.group(1)))
        elif m.group(2):
            tokens.append(("one", m.group(2)))
        elif m.group(3):
            tokens.append(("var", m.group(3)))
        elif m.group(4):
            sym = m.group(4)
            tokens.append(({"(": "lparen", ")": "rparen", ",": "comma"}[sym], sym))
    return tokens


def parse(src: str) -> Node:
    """Parse the textual form ``eml(a, b)`` / ``1`` / ``x`` into a :class:`Node`."""
    tokens = _tokenize(src)
    it = iter(tokens)
    node, leftover = _parse_expr(it, peek=next(it, None))
    if leftover is not None:
        raise ParseError(f"Unexpected trailing token: {leftover[1]!r}")
    return node


def _parse_expr(
    it: Iterator[Tuple[str, str]], peek: Tuple[str, str] | None
) -> tuple[Node, Tuple[str, str] | None]:
    if peek is None:
        raise ParseError("Unexpected end of input.")
    kind, text = peek
    if kind == "one":
        return One(), next(it, None)
    if kind == "var":
        return Var(text), next(it, None)
    if kind == "eml":
        nxt = next(it, None)
        if nxt is None or nxt[0] != "lparen":
            raise ParseError("Expected '(' after 'eml'.")
        left, peek2 = _parse_expr(it, next(it, None))
        if peek2 is None or peek2[0] != "comma":
            raise ParseError("Expected ',' between eml arguments.")
        right, peek3 = _parse_expr(it, next(it, None))
        if peek3 is None or peek3[0] != "rparen":
            raise ParseError("Expected ')' to close eml(...).")
        return Eml(left, right), next(it, None)
    raise ParseError(f"Unexpected token {text!r}.")


def parse_rpn(src: str) -> Node:
    """Parse the reverse-Polish form (space-separated tokens) into a :class:`Node`."""
    stack: list[Node] = []
    for tok in src.split():
        if tok == "1":
            stack.append(One())
        elif tok == "eml":
            if len(stack) < 2:
                raise ParseError("RPN stack underflow: 'eml' needs two operands.")
            right = stack.pop()
            left = stack.pop()
            stack.append(Eml(left, right))
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            stack.append(Var(tok))
        else:
            raise ParseError(f"Unrecognised RPN token: {tok!r}")
    if len(stack) != 1:
        raise ParseError(
            f"RPN did not reduce to a single expression (stack size {len(stack)})."
        )
    return stack[0]


__all__ = ["parse", "parse_rpn", "ParseError"]
