"""The exhaustive search is numeric: we require it to re-discover the known
small identities (exp at size 3, zero at size 7) without being told the answer.
"""

import cmath
import math

from eml.ast import tree_size
from eml.eval import evaluate
from eml.search import search_identity, verify_match


def test_search_rediscovers_exp():
    matches = search_identity(cmath.exp, variables=["x"], max_size=5, max_results=1)
    assert matches, "search failed to rediscover exp(x)"
    tree = matches[0]
    assert tree_size(tree) == 3
    # confirm it actually equals exp(x) at a fresh point
    for x in [0.2, 1.5, 3.1]:
        assert evaluate(tree, {"x": x}) == __pytest_approx(math.exp(x))


def test_search_rediscovers_zero():
    matches = search_identity(lambda: 0, variables=[], max_size=7, max_results=1)
    assert matches, "search failed to rediscover the constant 0"
    assert tree_size(matches[0]) == 7


def test_verify_match_positive():
    # exp(x) tree should verify against math.exp
    from eml.ast import Eml, One, Var

    tree = Eml(Var("x"), One())
    assert verify_match(tree, cmath.exp, ["x"])


def test_verify_match_negative():
    # exp(x) tree should NOT verify against cos
    from eml.ast import Eml, One, Var

    tree = Eml(Var("x"), One())
    assert not verify_match(tree, cmath.cos, ["x"])


def __pytest_approx(v):
    import pytest

    return pytest.approx(v, rel=1e-7, abs=1e-9)
