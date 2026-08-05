import math

import pytest

from eml.ast import Eml, One, Var
from eml.eval import eml_primitive, evaluate


def test_primitive_real():
    # eml(1, 1) = e - ln(1) = e
    assert eml_primitive(1, 1) == pytest.approx(math.e)
    # eml(0, 1) = 1 - 0 = 1
    assert eml_primitive(0, 1) == pytest.approx(1.0)


def test_primitive_complex_domain():
    # log(-1) = i*pi; eml(0, -1) = 1 - i*pi
    result = eml_primitive(0, -1)
    assert isinstance(result, complex)
    assert result.real == pytest.approx(1.0)
    assert result.imag == pytest.approx(-math.pi)


def test_evaluate_tree_with_binding():
    # exp(x) = eml(x, 1)
    t = Eml(Var("x"), One())
    assert evaluate(t, {"x": 2.0}) == pytest.approx(math.e**2)


def test_evaluate_missing_binding_raises():
    t = Eml(Var("x"), One())
    with pytest.raises(KeyError):
        evaluate(t)


def test_evaluate_constant_tree():
    # e = eml(1, 1)
    t = Eml(One(), One())
    assert evaluate(t) == pytest.approx(math.e)
