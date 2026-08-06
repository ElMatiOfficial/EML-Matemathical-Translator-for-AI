"""Numeric evaluation of EML trees.

The primitive is defined over the complex numbers using the principal
branch of the logarithm (matching the paper's convention):

    eml(x, y) = exp(x) - ln(y)

Evaluation accepts a mapping from variable names to numeric values
(``int``, ``float``, or ``complex``). Constants of type ``One`` always
evaluate to ``1``.

Extended-real arithmetic
------------------------
The paper's decompositions of ``neg``, ``add``, ``mul``, ``div``, ``pow``,
and most trigonometric identities use ``log(0) = -inf`` and
``exp(-inf) = 0`` as intermediate values. Python's built-in ``math``
module rejects both (``math.log(0)`` raises ``ValueError`` and
``math.exp(inf)`` raises ``OverflowError``). To make the paper's trees
evaluate end-to-end, this module extends ``eml_primitive`` with the
obvious limits:

* ``exp(-inf) = 0``,   ``exp(+inf) = +inf``
* ``log(0) = -inf``,   ``log(+inf) = +inf``
* ``log(y)`` for ``y < 0`` falls through to the complex principal branch
* indeterminate forms (``+inf - +inf``) return ``nan``
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Mapping
from typing import Union

from eml.ast import Eml, Node, One, Var

Number = Union[int, float, complex]

_NEG_INF = float("-inf")
_POS_INF = float("inf")
_NAN = float("nan")


def _safe_exp_real(x: float) -> float:
    if x == _NEG_INF:
        return 0.0
    if x == _POS_INF:
        return _POS_INF
    try:
        return math.exp(x)
    except OverflowError:
        return _POS_INF


def _safe_log_real(y: float) -> float | complex:
    """log(y) with the extended-real conventions described in the module docstring.

    Returns a ``complex`` when ``y`` is a negative finite real (principal branch);
    callers should then switch the whole computation to complex arithmetic.
    """
    if y == 0:
        return _NEG_INF
    if y == _POS_INF:
        return _POS_INF
    if y == _NEG_INF:
        # log(-inf) is +inf + i*pi on the principal branch
        return complex(_POS_INF, math.pi)
    if y < 0:
        return cmath.log(y)
    return math.log(y)


def eml_primitive(x: Number, y: Number) -> Number:
    """The EML primitive: ``exp(x) - ln(y)`` with extended-real handling."""
    if isinstance(x, complex) or isinstance(y, complex):
        return _eml_complex(x, y)

    # real path -- use the extended-real helpers
    a = _safe_exp_real(float(x))
    b = _safe_log_real(float(y))

    if isinstance(b, complex):
        # y was a negative finite real; promote the whole thing to complex
        return complex(a) - b

    # indeterminate forms
    if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
        return _NAN
    return a - b


def _eml_complex(x: Number, y: Number) -> complex:
    cx = complex(x)
    cy = complex(y)
    # complex log(0) is undefined -- return -inf (extended) on principal branch
    if cy == 0:
        # return the real -inf promoted to complex; callers can keep going
        return cmath.exp(cx) - complex(_NEG_INF)
    return cmath.exp(cx) - cmath.log(cy)


def evaluate(node: Node, bindings: Mapping[str, Number] | None = None) -> Number:
    """Evaluate an EML tree numerically.

    Parameters
    ----------
    node : Node
        The EML expression tree.
    bindings : mapping of variable name -> number, optional
        Values to substitute for each Var in the tree. Missing bindings
        raise ``KeyError``.
    """
    bindings = bindings or {}
    if isinstance(node, One):
        return 1
    if isinstance(node, Var):
        if node.name not in bindings:
            raise KeyError(
                f"No binding provided for variable {node.name!r}. "
                f"Pass bindings={{'{node.name}': value}}."
            )
        return bindings[node.name]
    if isinstance(node, Eml):
        x = evaluate(node.left, bindings)
        y = evaluate(node.right, bindings)
        return eml_primitive(x, y)
    raise TypeError(f"Unknown node type: {type(node).__name__}")
