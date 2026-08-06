"""Translate standard mathematics into EML trees.

The translator walks a :mod:`sympy` expression and rewrites each node using
the identities registered in :mod:`eml.identities`. It mirrors the
reference implementation from Odrzywolek's SymbolicRegressionPackage
(``eml_compiler_v4.py`` in https://github.com/VA00/SymbolicRegressionPackage ),
reusing its normalisation strategy but producing our AST directly rather
than string output.

Supported out of the box
------------------------
* Any :class:`sympy.Symbol`
* Integer and rational literals (built up from ``1`` via the paper's
  doubling recipe)
* ``E`` (Euler's number)
* ``exp(x)`` and ``log(x)``
* Arithmetic: unary negation, ``Add`` (including n-ary), ``Mul`` (n-ary),
  ``Pow`` with arbitrary base and exponent
* ``sqrt(x)``
* Trigonometric and hyperbolic functions via sympy's ``rewrite(exp)``
  normalisation (they lower through the same exp/log/pow chain)

Unsupported
-----------
Anything that sympy's ``rewrite`` chain cannot reduce to
exp / log / Pow / Add / Mul raises :class:`TranslationError`. In practice
this catches things like ``Abs``, special functions (Bessel, gamma),
piecewise expressions, and symbolic integrals.
"""

from __future__ import annotations

from typing import Union

import sympy

from eml.ast import Node, Var
from eml.identities import IDENTITIES, int_tree, rational_tree

ExprLike = Union[str, "sympy.Expr"]


class TranslationError(NotImplementedError):
    """Raised when a mathematical expression cannot be expressed as EML.

    This usually means sympy's ``rewrite`` chain failed to reduce the
    expression to a combination of Add/Mul/Pow/exp/log. Register an
    explicit identity with :func:`eml.register_identity`, or use
    :func:`eml.search_identity` to search for one.
    """


def _as_sympy(expr: ExprLike) -> sympy.Expr:
    if isinstance(expr, sympy.Expr):
        return expr
    return sympy.sympify(expr, rational=True)


def _normalize_to_exp_log(expr: sympy.Expr, max_iter: int = 8) -> sympy.Expr:
    """Iterated rewrite to reduce trig / hyperbolic / sec-csc-cot to exp/log/Pow."""
    e = expr
    for _ in range(max_iter):
        e2 = e.rewrite(sympy.log).rewrite(sympy.exp).rewrite(sympy.Pow)
        if e2 == e:
            break
        e = e2
    return e


def to_eml(expr: ExprLike) -> Node:
    """Translate a mathematical expression into an EML tree.

    Parameters
    ----------
    expr : sympy expression or string
        Standard mathematics. Strings are parsed with :func:`sympy.sympify`
        in rational mode (so ``0.5`` becomes ``1/2``, not ``5/10``).

    Returns
    -------
    Node
        The equivalent EML tree over the constant 1 and variable placeholders.

    Raises
    ------
    TranslationError
        When the expression references a function or constant that cannot
        be rewritten to Add/Mul/Pow/exp/log.
    """
    e = _as_sympy(expr)
    e = _normalize_to_exp_log(e)
    return _compile(e)


def _compile(expr: sympy.Expr) -> Node:
    """Recursive compilation. Mirrors upstream ``compile_to_eml``."""
    # --- atoms: numbers, symbols, named constants ---
    if expr.is_Atom:
        if expr is sympy.E or expr == sympy.E:
            return IDENTITIES["e"]()
        if expr is sympy.I or expr == sympy.I:
            return IDENTITIES["I"]()
        if expr == sympy.pi:
            return IDENTITIES["pi"]()
        if isinstance(expr, sympy.Integer):
            return int_tree(int(expr))
        if isinstance(expr, sympy.Rational):
            return rational_tree(int(expr.p), int(expr.q))
        if isinstance(expr, sympy.Float):
            # rationalise by going through the decimal string
            r = sympy.Rational(str(expr))
            return rational_tree(int(r.p), int(r.q))
        if isinstance(expr, sympy.Symbol):
            return Var(expr.name)
        raise TranslationError(f"Unsupported atom {expr!r} of type {type(expr).__name__}.")

    func = getattr(expr, "func", None)

    # --- exp / log ---
    if func is sympy.exp and len(expr.args) == 1:
        return IDENTITIES["exp"](_compile(expr.args[0]))
    if func is sympy.log:
        if len(expr.args) == 1:
            return IDENTITIES["ln"](_compile(expr.args[0]))
        # log(x, base) = ln(x) / ln(base)
        if len(expr.args) == 2:
            x_tree = IDENTITIES["ln"](_compile(expr.args[0]))
            b_tree = IDENTITIES["ln"](_compile(expr.args[1]))
            return IDENTITIES["div"](x_tree, b_tree)

    # --- Pow: a ** b ---
    if isinstance(expr, sympy.Pow):
        base, power = expr.as_base_exp()
        # sqrt is a common special case, keep the compact form
        if power == sympy.Rational(1, 2):
            return IDENTITIES["sqrt"](_compile(base))
        return IDENTITIES["pow"](_compile(base), _compile(power))

    # --- Mul: fold left with binary mul, handle -1 factor specially ---
    if isinstance(expr, sympy.Mul):
        factors = list(expr.args)
        # peel off any -1 so we emit a top-level neg
        negate = False
        peeled: list[sympy.Expr] = []
        for f in factors:
            if f == sympy.Integer(-1):
                negate = not negate
            else:
                peeled.append(f)
        if not peeled:
            return IDENTITIES["neg"](int_tree(1)) if negate else int_tree(1)
        acc = _compile(peeled[0])
        for g in peeled[1:]:
            acc = IDENTITIES["mul"](acc, _compile(g))
        return IDENTITIES["neg"](acc) if negate else acc

    # --- Add: fold left with binary add ---
    if isinstance(expr, sympy.Add):
        terms = list(expr.args)
        acc = _compile(terms[0])
        for t in terms[1:]:
            acc = IDENTITIES["add"](acc, _compile(t))
        return acc

    # --- last resort: try one more normalisation pass ---
    e2 = _normalize_to_exp_log(expr, max_iter=8)
    if e2 != expr:
        return _compile(e2)

    raise TranslationError(
        f"No EML decomposition known for {expr} (type {type(expr).__name__}). "
        "Register one with eml.register_identity or search for one with "
        "eml.search_identity."
    )


__all__ = ["to_eml", "TranslationError"]
