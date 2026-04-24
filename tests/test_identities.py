"""The identities module verifies each builder against a reference at import time.
Here we cross-check them with fresh sample points and confirm the public API.
"""

import cmath
import math

import pytest

from eml.ast import One, Var
from eml.eval import evaluate
from eml.identities import IDENTITIES, register_identity, verify_identity


@pytest.mark.parametrize("x", [0.5, 1.2, 2.7, 5.0])
def test_exp_identity(x):
    tree = IDENTITIES["exp"](Var("x"))
    assert evaluate(tree, {"x": x}) == pytest.approx(math.exp(x))


@pytest.mark.parametrize("x", [0.5, 1.2, 2.7, 5.0])
def test_ln_identity(x):
    tree = IDENTITIES["ln"](Var("x"))
    assert evaluate(tree, {"x": x}) == pytest.approx(math.log(x))


def test_e_identity():
    assert evaluate(IDENTITIES["e"]()) == pytest.approx(math.e)


def test_zero_identity():
    v = evaluate(IDENTITIES["zero"]())
    assert abs(complex(v)) < 1e-12


@pytest.mark.parametrize(
    "x,y", [(3.0, 1.5), (1.7, 0.6), (5.0, 2.0), (1.0, 0.8)]
)
def test_sub_identity(x, y):
    tree = IDENTITIES["sub"](Var("x"), Var("y"))
    assert evaluate(tree, {"x": x, "y": y}) == pytest.approx(x - y)


def test_register_and_reject_bogus_identity():
    # an identity that *claims* to compute cos(x) but actually computes exp(x)
    # should be rejected by verify_identity
    from eml.ast import Eml, One

    def fake_cos(x):
        return Eml(x, One())  # this is exp, not cos

    with pytest.raises(AssertionError):
        register_identity("cos", fake_cos, cmath.cos)


def test_register_valid_user_identity():
    """A user can register a derivation of their own, and it will be verified."""
    from eml.ast import Eml, Var

    # sinh + cosh = exp(x), i.e. sinh(x) = exp(x) - cosh(x). Instead we register
    # a tautological identity: exp(-x) computed as eml(-x_as_subtraction, 1)
    # Simplest test: an alias for exp
    def my_exp(x):
        return IDENTITIES["exp"](x)

    register_identity("my_exp", my_exp, math.exp)
    assert "my_exp" in IDENTITIES
    # cleanup so it doesn't leak into other tests
    IDENTITIES.pop("my_exp")


def test_verify_identity_public():
    # smoke test of the public verifier
    verify_identity(
        "sub",
        IDENTITIES["sub"],
        lambda x, y: x - y,
    )
