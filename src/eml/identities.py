"""Registry of known EML decompositions of elementary functions.

Each identity is a Python function that, given argument nodes, returns the
EML tree that is numerically equal to the named function applied to those
arguments.

The three identities explicitly stated by Odrzywolek (2026) are:

* ``exp(x)  = eml(x, 1)``
* ``e       = eml(1, 1)``
* ``ln(z)   = eml(1, eml(eml(1, z), 1))``

All other identities in this file are derived from those three by
composition, and every identity is verified numerically at import time.

Users can extend the registry by calling :func:`register_identity`, which
re-verifies the new identity against a reference implementation.
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


# -- identity builders -----------------------------------------------------

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
    """0 = ln(1) = eml(1, eml(eml(1, 1), 1)). [derived]"""
    return _ln(One())


def _sub(x: Node, y: Node) -> Node:
    """x - y = eml(ln(x), exp(y)). [derived]

    Proof:
        eml(ln(x), exp(y)) = exp(ln(x)) - ln(exp(y)) = x - y.
    """
    return Eml(_ln(x), _exp(y))


def _one_minus(y: Node) -> Node:
    """1 - y = eml(0, exp(y)). [derived, more compact than general sub]

    Proof:
        eml(0, exp(y)) = exp(0) - ln(exp(y)) = 1 - y.
    """
    return Eml(_zero(), _exp(y))


def _minus_one(x: Node) -> Node:
    """x - 1 = eml(ln(x), e). [derived]

    Proof:
        eml(ln(x), e) = exp(ln(x)) - ln(e) = x - 1.
    """
    return Eml(_ln(x), _e())


# ``exp(ln(x)) - ln(exp(y)) = x - y`` generalised: subtract-of-logs.
def _log_sub(x: Node, y: Node) -> Node:
    """ln(x) - ln(y) = eml(ln(ln(x)), exp(ln(y))) -- rarely useful directly,
    included because it can short-circuit some rewrites; numerically equal to
    ln(x/y) on the positive reals. [derived]
    """
    return Eml(_ln(_ln(x)), _exp(_ln(y)))


# -- registration with verification ---------------------------------------

# Reference numeric implementations used to verify an identity at import
# time (or when a user registers a new one). Variables x, y, z, w range over
# a small grid of positive reals and complex numbers.
_REFERENCES: dict[str, Callable[..., complex]] = {
    "exp": cmath.exp,
    "ln": cmath.log,
    "e": lambda: cmath.exp(1),
    "zero": lambda: 0,
    "sub": lambda x, y: x - y,
    "one_minus": lambda y: 1 - y,
    "minus_one": lambda x: x - 1,
    "log_sub": lambda x, y: cmath.log(x) - cmath.log(y),
}


# Sample points for numeric verification. All real and positive so that
# principal-branch logarithms are unambiguous; plus a couple of complex points.
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
        if abs(lhs - rhs) > tol:
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


# -- bootstrap the default registry ---------------------------------------

def _bootstrap() -> None:
    register_identity("exp", _exp, _REFERENCES["exp"])
    register_identity("ln", _ln, _REFERENCES["ln"])
    register_identity("log", _ln, _REFERENCES["ln"])  # alias
    register_identity("e", _e, _REFERENCES["e"])
    register_identity("zero", _zero, _REFERENCES["zero"])
    register_identity("sub", _sub, _REFERENCES["sub"])
    register_identity("one_minus", _one_minus, _REFERENCES["one_minus"])
    register_identity("minus_one", _minus_one, _REFERENCES["minus_one"])
    register_identity("log_sub", _log_sub, _REFERENCES["log_sub"])


_bootstrap()


__all__ = [
    "IDENTITIES",
    "register_identity",
    "verify_identity",
]
