from eml.ast import Eml, One, Var, free_vars, substitute, tree_depth, tree_size


def test_one_is_singleton_equal():
    assert One() == One()
    assert hash(One()) == hash(One())


def test_var_equality_by_name():
    assert Var("x") == Var("x")
    assert Var("x") != Var("y")


def test_tree_size_and_depth():
    # exp(x) = eml(x, 1)
    t = Eml(Var("x"), One())
    assert tree_size(t) == 3
    assert tree_depth(t) == 1

    # ln(x) = eml(1, eml(eml(1, x), 1))
    t = Eml(One(), Eml(Eml(One(), Var("x")), One()))
    assert tree_size(t) == 7
    assert tree_depth(t) == 3


def test_free_vars():
    t = Eml(Var("x"), Eml(Var("y"), One()))
    assert free_vars(t) == {"x", "y"}


def test_substitute():
    t = Eml(Var("x"), One())
    t2 = substitute(t, {"x": Eml(One(), One())})
    assert t2 == Eml(Eml(One(), One()), One())
