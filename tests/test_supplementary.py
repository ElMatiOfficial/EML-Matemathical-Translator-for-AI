"""Coverage for identities ported from the paper's SymbolicRegressionPackage.

Every identity is already numerically verified at import time by
``identities._bootstrap`` against its :data:`_REFERENCES` reference; these
tests re-verify on fresh sample points and confirm the headline
Kolmogorov-length claims from the paper hold in our AST.
"""

from __future__ import annotations

import cmath
import math

import pytest

import eml
from eml.ast import Var
from eml.eval import evaluate
from eml.identities import IDENTITIES, int_tree, rational_tree
from eml.printer import to_rpn


# -- new unary identities ------------------------------------------------

@pytest.mark.parametrize("x", [0.3, 1.7, 2.8])
def test_neg(x):
    tree = IDENTITIES["neg"](Var("x"))
    assert complex(evaluate(tree, {"x": x})).real == pytest.approx(-x)


@pytest.mark.parametrize("x", [0.25, 0.5, 2.0, 4.0])
def test_inv(x):
    tree = IDENTITIES["inv"](Var("x"))
    assert complex(evaluate(tree, {"x": x})).real == pytest.approx(1 / x)


@pytest.mark.parametrize("x", [0.25, 1.0, 9.0, 16.0])
def test_sqrt(x):
    tree = IDENTITIES["sqrt"](Var("x"))
    got = complex(evaluate(tree, {"x": x}))
    assert got.real == pytest.approx(math.sqrt(x))
    # imaginary part is floating-point noise through the complex chain
    assert abs(got.imag) < 1e-9


# -- new binary identities ----------------------------------------------

@pytest.mark.parametrize("x,y", [(1.5, 2.5), (3.0, 4.0), (0.5, 2.0)])
def test_add(x, y):
    tree = IDENTITIES["add"](Var("x"), Var("y"))
    assert complex(evaluate(tree, {"x": x, "y": y})).real == pytest.approx(x + y)


@pytest.mark.parametrize("x,y", [(2.0, 3.0), (1.5, 4.0), (0.7, 2.1)])
def test_mul(x, y):
    tree = IDENTITIES["mul"](Var("x"), Var("y"))
    assert complex(evaluate(tree, {"x": x, "y": y})).real == pytest.approx(x * y)


@pytest.mark.parametrize("x,y", [(6.0, 2.0), (10.0, 4.0), (1.0, 2.5)])
def test_div(x, y):
    tree = IDENTITIES["div"](Var("x"), Var("y"))
    assert complex(evaluate(tree, {"x": x, "y": y})).real == pytest.approx(x / y)


@pytest.mark.parametrize("a,b", [(2.0, 3.0), (4.0, 0.5), (1.5, 2.0)])
def test_pow(a, b):
    tree = IDENTITIES["pow"](Var("a"), Var("b"))
    got = complex(evaluate(tree, {"a": a, "b": b}))
    assert got.real == pytest.approx(a ** b)
    assert abs(got.imag) < 1e-9


# -- constants ----------------------------------------------------------

def test_two():
    tree = IDENTITIES["two"]()
    assert complex(evaluate(tree)).real == pytest.approx(2.0)


def test_I_is_positive_imaginary_unit():
    tree = IDENTITIES["I"]()
    got = complex(evaluate(tree))
    # our convention: +i (sympy/numpy-compatible)
    assert got.imag == pytest.approx(1.0, abs=1e-9)
    assert abs(got.real) < 1e-9


def test_pi():
    tree = IDENTITIES["pi"]()
    got = complex(evaluate(tree))
    assert got.real == pytest.approx(math.pi, rel=1e-10)
    assert abs(got.imag) < 1e-9


# -- Kolmogorov-length claims from the paper ----------------------------

def test_K_of_mul_is_41():
    """The paper states the Kolmogorov length of multiplication is 41."""
    tree = IDENTITIES["mul"](Var("x"), Var("y"))
    tokens = to_rpn(tree).split()
    assert len(tokens) == 41


def test_K_of_pi_is_193():
    """The paper states the Kolmogorov length of pi is 193."""
    tree = IDENTITIES["pi"]()
    tokens = to_rpn(tree).split()
    assert len(tokens) == 193


# -- integer and rational builders --------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10, -1, -7])
def test_int_tree(n):
    tree = int_tree(n)
    assert complex(evaluate(tree)).real == pytest.approx(float(n))


@pytest.mark.parametrize("p,q", [(1, 2), (3, 4), (-1, 3), (7, 1), (-5, 2)])
def test_rational_tree(p, q):
    tree = rational_tree(p, q)
    assert complex(evaluate(tree)).real == pytest.approx(p / q)


# -- forward translator end-to-end --------------------------------------

@pytest.mark.parametrize(
    "src, bindings, expected",
    [
        ("x + y", {"x": 3.0, "y": 4.0}, 7.0),
        ("x * y", {"x": 2.5, "y": 4.0}, 10.0),
        ("x / y", {"x": 10.0, "y": 4.0}, 2.5),
        ("-x", {"x": 3.3}, -3.3),
        ("x**2", {"x": 5.0}, 25.0),
        ("2*x + 1", {"x": 3.0}, 7.0),
        ("exp(x) - 1", {"x": 1.0}, math.e - 1),
        ("(x + 1) * (x - 1)", {"x": 3.0}, 8.0),
    ],
)
def test_forward_arithmetic(src, bindings, expected):
    tree = eml.to_eml(src)
    got = complex(evaluate(tree, bindings)).real
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "src, x, expected",
    [
        ("sin(x)", 0.5, math.sin(0.5)),
        ("cos(x)", 0.5, math.cos(0.5)),
        ("tan(x)", 0.5, math.tan(0.5)),
        ("sinh(x)", 0.7, math.sinh(0.7)),
        ("cosh(x)", 0.7, math.cosh(0.7)),
        ("tanh(x)", 0.7, math.tanh(0.7)),
    ],
)
def test_forward_trig(src, x, expected):
    tree = eml.to_eml(src)
    got = complex(evaluate(tree, {"x": x})).real
    assert got == pytest.approx(expected, rel=1e-8, abs=1e-9)


def test_forward_nested_polynomial():
    """Compile a messy polynomial and verify at several points."""
    tree = eml.to_eml("x**3 - 2*x**2 + x - 1")
    for x in [0.5, 1.0, 1.5, 2.0, 3.0]:
        expected = x ** 3 - 2 * x ** 2 + x - 1
        got = complex(evaluate(tree, {"x": x})).real
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)
