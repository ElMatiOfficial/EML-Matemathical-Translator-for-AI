"""Translate EML trees back into standard mathematical notation.

The inverse direction is conceptually simple: replace each
:class:`~eml.ast.Eml` node by the symbolic expression ``exp(L) - log(R)``
and let :mod:`sympy` simplify. Because the rewrite is a pure substitution
(no pattern matching or search), this direction is always exact.
"""

from __future__ import annotations

from typing import Mapping, Optional

import sympy

from eml.ast import Eml, Node, One, Var


def from_eml(
    node: Node,
    *,
    simplify: bool = True,
    symbols: Optional[Mapping[str, sympy.Symbol]] = None,
) -> sympy.Expr:
    """Translate an EML tree into the sympy expression it represents.

    Parameters
    ----------
    node : Node
        The EML tree to translate.
    simplify : bool, default True
        Run :func:`sympy.simplify` on the result. Recognises that e.g.
        ``eml(x, 1)`` is equivalent to ``exp(x)``.
    symbols : mapping, optional
        Explicit symbol assumptions (e.g. ``{"x": sympy.Symbol("x", positive=True)}``).
        When provided, simplification can fold ``log(exp(x))`` into ``x``.

    Returns
    -------
    sympy.Expr
    """
    symbols = dict(symbols) if symbols else {}
    raw = _to_sympy(node, symbols)
    if simplify:
        return sympy.simplify(raw)
    return raw


def _to_sympy(node: Node, symbols: dict[str, sympy.Symbol]) -> sympy.Expr:
    if isinstance(node, One):
        return sympy.Integer(1)
    if isinstance(node, Var):
        if node.name not in symbols:
            # sympy treats unconstrained symbols as complex by default; for
            # principal-branch cleanup we default to a real positive symbol,
            # which is the domain the paper's identities were verified on.
            symbols[node.name] = sympy.Symbol(node.name, positive=True)
        return symbols[node.name]
    if isinstance(node, Eml):
        x = _to_sympy(node.left, symbols)
        y = _to_sympy(node.right, symbols)
        return sympy.exp(x) - sympy.log(y)
    raise TypeError(f"Unknown node type: {type(node).__name__}")


__all__ = ["from_eml"]
