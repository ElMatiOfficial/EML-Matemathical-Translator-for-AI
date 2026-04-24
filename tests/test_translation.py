import math

import pytest
import sympy

from eml.eval import evaluate
from eml.forward import TranslationError, to_eml
from eml.inverse import from_eml


# -- forward: math -> EML -------------------------------------------------

@pytest.mark.parametrize(
    "src, value_at_x3",
    [
        ("exp(x)", math.exp(3.0)),
        ("log(x)", math.log(3.0)),
        ("E", math.e),
        ("1", 1),
    ],
)
def test_to_eml_basic(src, value_at_x3):
    tree = to_eml(src)
    bindings = {"x": 3.0} if "x" in src else {}
    assert complex(evaluate(tree, bindings)).real == pytest.approx(value_at_x3)


@pytest.mark.parametrize(
    "src, vars_, expected",
    [
        ("x - y", {"x": 5.0, "y": 2.0}, 3.0),
        ("1 - x", {"x": 0.7}, 0.3),
        ("x - 1", {"x": 4.2}, 3.2),
        ("exp(x) - 1", {"x": 1.0}, math.e - 1),
    ],
)
def test_to_eml_subtraction_forms(src, vars_, expected):
    tree = to_eml(src)
    got = complex(evaluate(tree, vars_)).real
    assert got == pytest.approx(expected)


def test_to_eml_unknown_function_errors():
    """A genuinely unknown atom (e.g. Abs) cannot be rewritten to exp/log."""
    with pytest.raises(TranslationError):
        to_eml("Abs(x)")


# -- inverse: EML -> math ------------------------------------------------

@pytest.mark.parametrize("src", ["exp(x)", "log(x)", "E", "1"])
def test_roundtrip_to_eml_and_back(src):
    """Forward then inverse should yield an expression numerically equal to the original.

    We compare numerically at sample points rather than symbolically, because
    ``from_eml`` annotates its symbols as positive (enabling stronger simplify),
    which makes direct symbolic subtraction against externally-created symbols
    produce a nonzero-looking expression even when the two are mathematically equal.
    """
    tree = to_eml(src)
    back = from_eml(tree)
    expr = sympy.sympify(src)

    if expr.free_symbols:
        # evaluate at positive reals where both sides are defined
        sym_back = next(iter(back.free_symbols))
        sym_expr = next(iter(expr.free_symbols))
        for val in [0.5, 1.3, 3.7]:
            lhs = complex(back.subs(sym_back, val))
            rhs = complex(expr.subs(sym_expr, val))
            assert abs(lhs - rhs) < 1e-10
    else:
        assert complex(back) == pytest.approx(complex(expr))


def test_from_eml_literal_tree():
    from eml.ast import Eml, One, Var

    tree = Eml(Var("x"), One())
    got = from_eml(tree)
    # numeric check: exp(2) ~ 7.389
    sym = next(iter(got.free_symbols))
    assert complex(got.subs(sym, 2.0)).real == pytest.approx(math.e ** 2)
