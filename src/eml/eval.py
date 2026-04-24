"""Numeric evaluation of EML trees.

The primitive is defined over the complex numbers using the principal
branch of the logarithm (matching the paper's convention):

    eml(x, y) = exp(x) - ln(y)

Evaluation accepts a mapping from variable names to numeric values
(``int``, ``float``, or ``complex``). Constants of type ``One`` always
evaluate to ``1``.
"""

from __future__ import annotations

import cmath
import math
from typing import Mapping, Union

from eml.ast import Eml, Node, One, Var

Number = Union[int, float, complex]


def eml_primitive(x: Number, y: Number) -> Number:
    """The EML primitive: exp(x) - ln(y).

    If either input is complex, the whole computation is performed in the
    complex domain using the principal branch of ``ln``. Otherwise a real
    result is returned when possible.
    """
    if isinstance(x, complex) or isinstance(y, complex):
        return cmath.exp(x) - cmath.log(y)
    # real fast path
    try:
        return math.exp(x) - math.log(y)
    except (ValueError, OverflowError):
        # fall back to complex for negative/zero logarithms, large exponents, etc.
        return cmath.exp(x) - cmath.log(y)


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
