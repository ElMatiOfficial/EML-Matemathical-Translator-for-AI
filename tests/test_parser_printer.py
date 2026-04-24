import pytest

from eml.ast import Eml, One, Var
from eml.parser import ParseError, parse, parse_rpn
from eml.printer import to_latex, to_rpn, to_text, to_tree


def _roundtrip(tree, parser, printer):
    assert parser(printer(tree)) == tree


def test_text_roundtrip_leaves():
    for t in [One(), Var("x")]:
        _roundtrip(t, parse, to_text)


def test_text_roundtrip_nested():
    t = Eml(Var("x"), Eml(One(), Var("y")))
    _roundtrip(t, parse, to_text)


def test_rpn_roundtrip():
    t = Eml(One(), Eml(Eml(One(), Var("x")), One()))
    _roundtrip(t, parse_rpn, to_rpn)


def test_rpn_k_length_of_exp_is_3():
    # exp(x) = eml(x, 1) should have K=3 in RPN (matches paper)
    t = Eml(Var("x"), One())
    tokens = to_rpn(t).split()
    assert len(tokens) == 3


def test_rpn_k_length_of_ln_is_7():
    # ln(x) should have K=7 in RPN (matches paper)
    t = Eml(One(), Eml(Eml(One(), Var("x")), One()))
    tokens = to_rpn(t).split()
    assert len(tokens) == 7


def test_latex_contains_operatorname():
    t = Eml(Var("x"), One())
    assert r"\operatorname{eml}" in to_latex(t)


def test_tree_ascii_is_pure_ascii():
    t = Eml(One(), Eml(Eml(One(), Var("x")), One()))
    out = to_tree(t)
    assert out.encode("ascii")  # must round-trip through ASCII


def test_parse_rejects_bare_numbers():
    # Only the literal 1 is legal in pure EML; 2, 3, etc. must be built up.
    with pytest.raises(ParseError):
        parse("2")


def test_parse_error_on_missing_paren():
    with pytest.raises(ParseError):
        parse("eml(x, 1")
