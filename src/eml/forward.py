"""Translate standard mathematics into EML trees.

The translator walks a :mod:`sympy` expression and rewrites each node using
the identities registered in :mod:`eml.identities`.

Supported out of the box
------------------------
* Variables (any :class:`sympy.Symbol`)
* The literals ``0``, ``1``
* ``exp(x)``, ``log(x)`` / ``ln(x)``, ``E`` (Euler's number)
* Subtraction ``x - y`` (including the special forms ``1 - y`` and ``x - 1``)

Extending
---------
Any function or constant that does not have a registered identity raises
:class:`TranslationError`. Two paths forward:

1. Register a derivation via :func:`eml.register_identity` with a numeric
   reference. This is the right path for identities you know in closed form.
2. Discover one by exhaustive search with :func:`eml.search_identity`.

This design is faithful to the paper: Odrzywolek (2026) shows that EML
decompositions *exist* for every elementary function, but the concrete
trees for many of them -- addition, multiplication, sin, cos, sqrt, etc. --
have large Kolmogorov length (multiplication alone is K=41, pi is K=193)
and are the product of exhaustive search rather than simple derivation.
"""

from __future__ import annotations

from typing import Union

import sympy

from eml.ast import Eml, Node, One, Var
from eml.identities import IDENTITIES

ExprLike = Union[str, "sympy.Expr"]


class TranslationError(NotImplementedError):
    """Raised when a mathematical expression cannot (yet) be expressed as EML.

    The message will include a hint pointing at either
    :func:`eml.register_identity` or :func:`eml.search_identity`.
    """


def _as_sympy(expr: ExprLike) -> sympy.Expr:
    if isinstance(expr, sympy.Expr):
        return expr
    return sympy.sympify(expr)


def to_eml(expr: ExprLike) -> Node:
    """Translate a mathematical expression into an EML tree.

    Parameters
    ----------
    expr : sympy expression or string
        Standard mathematics. Strings are parsed with :func:`sympy.sympify`.

    Returns
    -------
    Node
        The equivalent EML tree over the constant 1 and variable placeholders.

    Raises
    ------
    TranslationError
        When the expression references a function or constant that has no
        registered EML identity.
    """
    return _translate(_as_sympy(expr))


def _translate(e: sympy.Expr) -> Node:
    # --- variables ---
    if isinstance(e, sympy.Symbol):
        return Var(e.name)

    # --- numeric atoms ---
    if e == sympy.Integer(1):
        return One()
    if e == sympy.Integer(0):
        return IDENTITIES["zero"]()

    # --- Euler's number ---
    if e is sympy.E or e == sympy.E:
        return IDENTITIES["e"]()

    # --- exponentiation via exp(...) / E**... ---
    if isinstance(e, sympy.exp):
        return IDENTITIES["exp"](_translate(e.args[0]))

    # --- logarithm ---
    if isinstance(e, sympy.log):
        if len(e.args) == 1:
            return IDENTITIES["ln"](_translate(e.args[0]))
        # log(x, base) = ln(x) / ln(base) -- needs division, not registered yet
        raise TranslationError(
            "log(x, base) requires a division identity, which is not yet registered. "
            "Use eml.search_identity to find a decomposition or register one manually."
        )

    # --- subtraction: sympy models a - b as Add(a, Mul(-1, b)) ---
    if isinstance(e, sympy.Add):
        return _translate_add(e)

    # --- E**x (equivalent to exp(x)) ---
    if isinstance(e, sympy.Pow):
        base, exponent = e.args
        if base == sympy.E:
            return IDENTITIES["exp"](_translate(exponent))
        raise TranslationError(
            f"Power {e} has no registered EML identity. Only E**x (== exp(x)) is "
            "built in. Register a decomposition or use search_identity."
        )

    # --- unsupported ---
    raise TranslationError(
        f"No EML identity registered for {type(e).__name__} in expression {e}. "
        "Register one with eml.register_identity or discover one with eml.search_identity."
    )


def _translate_add(e: sympy.Add) -> Node:
    """Translate an :class:`sympy.Add` that is exactly a binary subtraction.

    Handles:
        x - y, 1 - y, x - 1
    where one operand is positive-sign and the other is a -1 * something.
    Genuine addition is NOT supported yet (no registered identity).
    """
    args = list(e.args)
    if len(args) != 2:
        raise TranslationError(
            "Addition of more than two terms has no registered EML identity. "
            "Rewrite as a pair of subtractions or register an identity."
        )
    # Identify positive and negated arguments
    pos, neg = None, None
    for a in args:
        # detect -something (Mul(-1, x) or an explicit negative number)
        coeff = a.as_coeff_Mul()[0] if a.is_Mul else None
        if coeff is not None and coeff.is_negative:
            neg = -a  # strip the -1
        elif a.is_number and a.is_negative:
            neg = -a
        else:
            pos = a
    if pos is None or neg is None:
        raise TranslationError(
            f"Addition {e} is not of the form x - y and has no registered EML "
            "identity. Register one with eml.register_identity."
        )
    # Special compact forms first
    if pos == sympy.Integer(1):
        return IDENTITIES["one_minus"](_translate(neg))
    if neg == sympy.Integer(1):
        return IDENTITIES["minus_one"](_translate(pos))
    return IDENTITIES["sub"](_translate(pos), _translate(neg))


__all__ = ["to_eml", "TranslationError"]
