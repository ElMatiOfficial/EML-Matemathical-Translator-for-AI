"""Basic usage of the EML translator.

Run with:

    PYTHONPATH=src python examples/basic_usage.py
"""

from __future__ import annotations

import math

import eml


def demo_forward() -> None:
    print("=" * 60)
    print("Forward translation: standard math -> EML")
    print("=" * 60)
    for src in ["exp(x)", "log(x)", "E", "0", "1", "x - y", "exp(x) - 1"]:
        try:
            tree = eml.to_eml(src)
        except eml.TranslationError as err:  # type: ignore[attr-defined]
            print(f"  {src:14s} -> (no identity registered) {err}")
            continue
        size = eml.tree_size(tree)
        k = len(eml.to_rpn(tree).split())
        print(f"  {src:14s} -> {eml.to_text(tree):70s} size={size}, K={k}")


def demo_inverse() -> None:
    print()
    print("=" * 60)
    print("Inverse translation: EML -> standard math")
    print("=" * 60)
    for src in [
        "eml(x, 1)",
        "eml(1, 1)",
        "eml(1, eml(eml(1, x), 1))",
        "eml(eml(1, eml(eml(1, x), 1)), eml(y, 1))",
    ]:
        tree = eml.parse(src)
        print(f"  {src:45s} -> {eml.from_eml(tree)}")


def demo_numeric() -> None:
    print()
    print("=" * 60)
    print("Numeric sanity check")
    print("=" * 60)
    for src, bindings, ref in [
        ("exp(x)", {"x": 2.0}, math.exp(2.0)),
        ("log(x)", {"x": 5.0}, math.log(5.0)),
        ("x - y", {"x": 7.0, "y": 3.5}, 3.5),
    ]:
        tree = eml.to_eml(src)
        lhs = complex(eml.evaluate(tree, bindings)).real
        print(f"  {src:12s} {bindings}  -> {lhs:.10f}  (ref {ref:.10f})")


def demo_search() -> None:
    print()
    print("=" * 60)
    print("Exhaustive search rediscovers small identities")
    print("=" * 60)
    import cmath

    for name, fn, nvars in [
        ("exp(x)", cmath.exp, 1),
        ("0", lambda: 0, 0),
    ]:
        variables = ["x"] if nvars == 1 else []
        matches = eml.search_identity(fn, variables=variables, max_size=7, max_results=1)
        tree = matches[0] if matches else None
        if tree:
            print(f"  target {name:8s} found at size {eml.tree_size(tree)}: {eml.to_text(tree)}")
        else:
            print(f"  target {name:8s}: no match within budget")


def demo_custom_identity() -> None:
    print()
    print("=" * 60)
    print("Register a custom identity (cosh-style example)")
    print("=" * 60)
    import cmath

    from eml.ast import Eml, One, Var

    # An extra alias so the forward translator can see a new name.
    # (Register something we can actually derive: e^(x+0) = e^x.)
    def exp_plus_zero(x):
        return Eml(Eml(x, One()), Eml(One(), Eml(Eml(One(), One()), One())))

    # Numerically this is exp(x) - ln(ln(1)) ... not equal to exp(x).
    # Here we show that verification CATCHES the bad identity:
    try:
        eml.register_identity("bad_exp", exp_plus_zero, reference=cmath.exp)
        print("  (verification failed to catch a bad identity!)")
    except AssertionError as err:
        print(f"  bad identity rejected (good): {type(err).__name__}")


if __name__ == "__main__":
    demo_forward()
    demo_inverse()
    demo_numeric()
    demo_search()
    demo_custom_identity()
