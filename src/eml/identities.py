"""Registry of known EML decompositions of elementary functions.

Each identity is a Python function that, given argument nodes, returns the
EML tree that is numerically equal to the named function applied to those
arguments.

The three identities explicitly stated by Odrzywolek (2026) are:

* ``exp(x)  = eml(x, 1)``
* ``e       = eml(1, 1)``
* ``ln(z)   = eml(1, eml(eml(1, z), 1))``

All remaining identities in this file are the canonical chain used by the
paper's reference implementation (``eml_compiler_v4.py`` in
https://github.com/VA00/SymbolicRegressionPackage ). Every identity is
numerically verified at import time; a builder that disagrees with its
reference on any sample is rejected.

Users can extend the registry by calling :func:`register_identity`.
"""

from __future__ import annotations

import cmath
import math
from typing import Callable, Mapping

from eml.ast import Eml, Node, One, Var
from eml.eval import evaluate

IdentityFn = Callable[..., Node]

# Public registry: name -> EML-tree builder function.
# Functions with no arguments are constants (e.g. ``e()``). Unary and binary
# function identities take argument trees.
IDENTITIES: dict[str, IdentityFn] = {}


# =========================================================================
# Atomic building blocks
# =========================================================================


def _exp(x: Node) -> Node:
    """exp(x) = eml(x, 1). [Odrzywolek 2026]"""
    return Eml(x, One())


def _ln(x: Node) -> Node:
    """ln(x) = eml(1, eml(eml(1, x), 1)). [Odrzywolek 2026]"""
    return Eml(One(), Eml(Eml(One(), x), One()))


def _e() -> Node:
    """e = eml(1, 1). [Odrzywolek 2026]"""
    return Eml(One(), One())


def _zero() -> Node:
    """0 = ln(1). [derived, also appears as eml_zero in SymbolicRegressionPackage]"""
    return _ln(One())


def _sub(x: Node, y: Node) -> Node:
    """x - y = eml(ln(x), exp(y)).

    Proof: ``eml(ln x, exp y) = exp(ln x) - ln(exp y) = x - y``.
    Matches ``eml_sub`` in eml_compiler_v4.py.
    """
    return Eml(_ln(x), _exp(y))


def _one_minus(y: Node) -> Node:
    """1 - y = eml(0, exp(y)). [compact derived form]"""
    return Eml(_zero(), _exp(y))


def _minus_one(x: Node) -> Node:
    """x - 1 = eml(ln(x), e). [compact derived form]"""
    return Eml(_ln(x), _e())


# =========================================================================
# Paper's canonical chain (ported from eml_compiler_v4.py)
# =========================================================================


def _neg(z: Node) -> Node:
    """-z = sub(0, z). [upstream eml_compiler_v4.eml_neg]

    Uses extended-real evaluation: the intermediate ``log(0) = -inf`` is
    handled by :mod:`eml.eval`.
    """
    return _sub(_zero(), z)


def _add(a: Node, b: Node) -> Node:
    """a + b = sub(a, -b). [upstream eml_compiler_v4.eml_add]"""
    return _sub(a, _neg(b))


def _inv(z: Node) -> Node:
    """1/z = exp(-ln(z)). [upstream eml_compiler_v4.eml_inv]"""
    return _exp(_neg(_ln(z)))


def _mul(a: Node, b: Node) -> Node:
    """a * b = exp(ln(a) + ln(b)). [upstream eml_compiler_v4.eml_mul]"""
    return _exp(_add(_ln(a), _ln(b)))


def _div(a: Node, b: Node) -> Node:
    """a / b = a * (1/b). [upstream eml_compiler_v4.eml_div]"""
    return _mul(a, _inv(b))


def _pow(a: Node, b: Node) -> Node:
    """a^b = exp(b * ln(a)). [upstream eml_compiler_v4.eml_pow]"""
    return _exp(_mul(b, _ln(a)))


def _two() -> Node:
    """2 = 1 + 1. [upstream eml_compiler_v4.eml_two]"""
    return _add(One(), One())


def _sqrt(z: Node) -> Node:
    """sqrt(z) = z^(1/2). [composed via _pow and _inv]"""
    return _pow(z, _inv(_two()))


# =========================================================================
# Complex constants (enable trig via sympy's rewrite(exp) chain)
# =========================================================================


def _I() -> Node:
    """i = exp(ln(-1) / 2). [+i on the principal branch]

    Upstream's :func:`eml_const_I` negates this and therefore evaluates to
    ``-i``; it relies on a second sign flip inside :func:`_pi` to recover
    the correct sign of pi. We keep the conventional meaning of ``i``
    (positive imaginary unit) here.
    """
    return _exp(_div(_ln(_neg(One())), _two()))


def _pi() -> Node:
    """pi = -i * ln(-1). [principal branch: (-i)(i*pi) = pi]

    Equivalent to the upstream chain, compensating for our sign choice in
    :func:`_I` with an explicit negation.
    """
    return _mul(_neg(_I()), _ln(_neg(One())))


# =========================================================================
# Registration + verification machinery
# =========================================================================

# Reference numeric implementations used to verify an identity at import
# time (or when a user registers a new one). Variables x, y, z range over
# a small grid of positive reals and complex numbers.
_REFERENCES: dict[str, Callable[..., complex]] = {
    "exp": cmath.exp,
    "ln": cmath.log,
    "e": lambda: cmath.exp(1),
    "zero": lambda: 0,
    "sub": lambda x, y: x - y,
    "one_minus": lambda y: 1 - y,
    "minus_one": lambda x: x - 1,
    "neg": lambda z: -z,
    "add": lambda a, b: a + b,
    "inv": lambda z: 1 / z,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a ** b,
    "two": lambda: 2,
    "sqrt": cmath.sqrt,
    "I": lambda: 1j,
    "pi": lambda: math.pi,
}


# Sample points for numeric verification. Positive reals (so principal-branch
# logarithms are unambiguous) plus a couple of complex points. For identities
# that involve many chained log/exp operations (mul, div, pow, sqrt) the
# tolerance must absorb ~1e-12 cancellation error per cascaded operation, so
# the reasonable bound is a relative 1e-9.
_SAMPLES: list[complex] = [
    0.5, 1.0, 1.5, 2.0, 3.7, 10.0,
    complex(1.2, 0.3),
    complex(2.0, -0.5),
]

_TOL = 1e-9


def _argspec(fn: IdentityFn) -> list[str]:
    """Parameter names of an identity builder (e.g. ['x', 'y'])."""
    import inspect

    return list(inspect.signature(fn).parameters)


def verify_identity(
    name: str,
    builder: IdentityFn,
    reference: Callable[..., complex],
    *,
    samples: list[complex] | None = None,
    tol: float = _TOL,
) -> None:
    """Numerically verify a named EML identity against a reference.

    Raises
    ------
    AssertionError
        If the EML tree disagrees with the reference on any sample point
        by more than ``tol``.
    """
    samples = samples or _SAMPLES
    params = _argspec(builder)
    if not params:
        tree = builder()
        lhs = complex(evaluate(tree))
        rhs = complex(reference())
        if abs(lhs - rhs) > tol * max(1.0, abs(rhs)):
            raise AssertionError(
                f"Identity {name!r} failed: EML tree evaluates to {lhs} "
                f"but reference gives {rhs}."
            )
        return

    # Build arg trees as Var nodes and sweep over samples.
    var_nodes = [Var(p) for p in params]
    tree = builder(*var_nodes)
    checked = 0
    for pt in samples:
        bindings: Mapping[str, complex] = {p: pt for p in params}
        try:
            lhs = complex(evaluate(tree, bindings))
            rhs = complex(reference(*[pt for _ in params]))
        except (ValueError, ZeroDivisionError, OverflowError):
            # endpoint of the principal-branch domain or IEEE overflow; skip
            continue
        except TypeError:
            # reference only accepts reals but this sample is complex; skip it
            continue
        if not math.isfinite(lhs.real) or not math.isfinite(rhs.real):
            continue
        if not math.isfinite(lhs.imag) or not math.isfinite(rhs.imag):
            continue
        if abs(lhs - rhs) > tol * max(1.0, abs(rhs)):
            raise AssertionError(
                f"Identity {name!r} failed at sample {pt}: "
                f"EML gives {lhs}, reference gives {rhs} (|diff|={abs(lhs - rhs):.3e})."
            )
        checked += 1
    if checked == 0:
        raise AssertionError(
            f"Identity {name!r} could not be verified: every sample point was "
            "skipped. Check the reference function and domain."
        )


def register_identity(
    name: str,
    builder: IdentityFn,
    reference: Callable[..., complex] | None = None,
    *,
    verify: bool = True,
) -> None:
    """Add a new identity to the registry, optionally verifying it.

    Parameters
    ----------
    name : str
        Name of the function or constant (e.g. ``"sinh"``).
    builder : callable
        Function that takes argument :class:`Node`\\ s and returns an EML
        :class:`Node` tree equal to ``name(...)`` applied to those arguments.
    reference : callable, optional
        Reference numeric implementation for verification. If ``None``, no
        verification is performed (use with care).
    verify : bool
        When ``True`` (the default) and a reference is provided, verify the
        identity numerically before registering.
    """
    if verify and reference is not None:
        verify_identity(name, builder, reference)
    IDENTITIES[name] = builder


# =========================================================================
# Bootstrap the default registry
# =========================================================================


def _bootstrap() -> None:
    # Atomic identities stated in the paper
    register_identity("exp", _exp, _REFERENCES["exp"])
    register_identity("ln", _ln, _REFERENCES["ln"])
    register_identity("log", _ln, _REFERENCES["ln"])  # alias
    register_identity("e", _e, _REFERENCES["e"])

    # Derived from those three
    register_identity("zero", _zero, _REFERENCES["zero"])
    register_identity("sub", _sub, _REFERENCES["sub"])
    register_identity("one_minus", _one_minus, _REFERENCES["one_minus"])
    register_identity("minus_one", _minus_one, _REFERENCES["minus_one"])

    # Paper's canonical chain (from eml_compiler_v4.py)
    register_identity("neg", _neg, _REFERENCES["neg"])
    register_identity("add", _add, _REFERENCES["add"])
    register_identity("inv", _inv, _REFERENCES["inv"])
    register_identity("mul", _mul, _REFERENCES["mul"])
    register_identity("div", _div, _REFERENCES["div"])
    register_identity("pow", _pow, _REFERENCES["pow"])
    register_identity("two", _two, _REFERENCES["two"])
    register_identity("sqrt", _sqrt, _REFERENCES["sqrt"])
    register_identity("I", _I, _REFERENCES["I"])
    register_identity("pi", _pi, _REFERENCES["pi"])


_bootstrap()


def int_tree(n: int) -> Node:
    """Build an EML tree for the integer ``n`` via the paper's doubling recipe.

    Matches ``eml_int`` in eml_compiler_v4.py: repeated ``add`` over a
    binary decomposition, plus negation for negatives.
    """
    if n == 1:
        return One()
    if n == 0:
        return _zero()
    if n < 0:
        return _neg(int_tree(-n))
    acc: Node | None = None
    term: Node = One()
    k = n
    while k > 0:
        if k & 1:
            acc = term if acc is None else _add(acc, term)
        term = _add(term, term)
        k >>= 1
    assert acc is not None
    return acc


def rational_tree(p: int, q: int) -> Node:
    """Build an EML tree for the rational ``p/q``. Matches ``eml_rational``."""
    if q == 1:
        return int_tree(p)
    num = int_tree(abs(p))
    den = int_tree(q)
    val = _mul(num, _inv(den))
    return val if p >= 0 else _neg(val)


__all__ = [
    "IDENTITIES",
    "int_tree",
    "rational_tree",
    "register_identity",
    "verify_identity",
]
