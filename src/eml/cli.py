"""Command-line interface for the EML translator.

Usage examples::

    eml translate "exp(x) - 1"
    eml reverse   "eml(x, 1)"
    eml identities
    eml verify    "eml(x, 1)" --as "exp(x)"
    eml search    "sin(x)" --vars x --max-size 9
    eml render    "eml(x, 1)" --format latex
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from eml import (
    IDENTITIES,
    evaluate,
    from_eml,
    parse,
    parse_rpn,
    to_eml,
    to_latex,
    to_rpn,
    to_text,
    to_tree,
)
from eml.search import search_identity, verify_match


def _parse_expr(src: str):
    """Accept either textual or RPN form; prefer textual."""
    if "(" in src:
        return parse(src)
    return parse_rpn(src)


def _cmd_translate(args: argparse.Namespace) -> int:
    tree = to_eml(args.expr)
    print(_render(tree, args.format))
    return 0


def _cmd_reverse(args: argparse.Namespace) -> int:
    tree = _parse_expr(args.eml_expr)
    result = from_eml(tree, simplify=not args.no_simplify)
    print(result)
    return 0


def _cmd_identities(args: argparse.Namespace) -> int:
    from eml.ast import Var

    for name, builder in sorted(IDENTITIES.items()):
        import inspect

        params = list(inspect.signature(builder).parameters)
        arg_trees = [Var(p) for p in params]
        tree = builder(*arg_trees)
        lhs = f"{name}({', '.join(params)})" if params else name
        print(f"{lhs:14s} = {to_text(tree)}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    tree = _parse_expr(args.eml_expr)
    import sympy

    target_expr = sympy.sympify(args.equivalent_to)
    # simple numeric sanity check using shared sampling
    variables = sorted({s.name for s in target_expr.free_symbols})
    if not variables:
        lhs = complex(evaluate(tree))
        rhs = complex(target_expr)
        ok = abs(lhs - rhs) < 1e-9
        print(f"LHS  = {lhs}")
        print(f"RHS  = {rhs}")
        print(f"match: {ok}")
        return 0 if ok else 1
    target_fn = sympy.lambdify(variables, target_expr, modules="cmath")

    ok = verify_match(tree, target_fn, variables)
    print(f"tree   : {to_text(tree)}")
    print(f"equals : {target_expr}")
    print(f"match  : {ok}")
    return 0 if ok else 1


def _cmd_search(args: argparse.Namespace) -> int:
    import sympy

    expr = sympy.sympify(args.target)
    variables = list(args.vars) or sorted({s.name for s in expr.free_symbols})
    target_fn = sympy.lambdify(variables, expr, modules="cmath")
    print(
        f"Searching for EML decompositions of {expr} over variables {variables} "
        f"(max_size={args.max_size})...",
        file=sys.stderr,
    )
    matches = search_identity(
        target_fn,
        variables=variables,
        max_size=args.max_size,
        tol=args.tol,
        max_results=args.max_results,
        verbose=args.verbose,
    )
    if not matches:
        print("No matching EML tree found within the size budget.", file=sys.stderr)
        return 1
    for i, t in enumerate(matches, 1):
        print(f"[{i}] size={_size(t)}  {_render(t, args.format)}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    tree = _parse_expr(args.eml_expr)
    print(_render(tree, args.format))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    tree = _parse_expr(args.eml_expr)
    bindings: dict[str, complex] = {}
    for item in args.binding:
        if "=" not in item:
            print(f"invalid --bind entry {item!r}, expected NAME=VALUE", file=sys.stderr)
            return 2
        name, val = item.split("=", 1)
        try:
            bindings[name.strip()] = float(val)
        except ValueError:
            bindings[name.strip()] = complex(val)
    result = evaluate(tree, bindings)
    print(result)
    return 0


def _render(tree, fmt: str) -> str:
    if fmt == "text":
        return to_text(tree)
    if fmt == "rpn":
        return to_rpn(tree)
    if fmt == "latex":
        return to_latex(tree)
    if fmt == "tree":
        return to_tree(tree)
    raise ValueError(f"unknown format {fmt!r}")


def _size(tree) -> int:
    from eml.ast import tree_size

    return tree_size(tree)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eml",
        description="Translate between standard mathematics and EML (Exp-Minus-Log).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="standard math -> EML tree")
    t.add_argument("expr", help="math expression, e.g. 'exp(x) - 1'")
    t.add_argument("--format", choices=["text", "rpn", "latex", "tree"], default="text")
    t.set_defaults(func=_cmd_translate)

    r = sub.add_parser("reverse", help="EML tree -> standard math")
    r.add_argument("eml_expr", help="EML in textual or RPN form")
    r.add_argument("--no-simplify", action="store_true")
    r.set_defaults(func=_cmd_reverse)

    i = sub.add_parser("identities", help="list registered EML identities")
    i.set_defaults(func=_cmd_identities)

    v = sub.add_parser("verify", help="check an EML tree equals a math expression")
    v.add_argument("eml_expr")
    v.add_argument(
        "--as", dest="equivalent_to", required=True, help="the math expression to check against"
    )
    v.set_defaults(func=_cmd_verify)

    s = sub.add_parser("search", help="exhaustive search for a new EML identity")
    s.add_argument("target", help="target math expression, e.g. 'sin(x)'")
    s.add_argument("--vars", nargs="*", default=[], help="variable names (default: inferred)")
    s.add_argument("--max-size", type=int, default=7)
    s.add_argument("--tol", type=float, default=1e-8)
    s.add_argument("--max-results", type=int, default=3)
    s.add_argument("--format", choices=["text", "rpn", "latex", "tree"], default="text")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=_cmd_search)

    rn = sub.add_parser("render", help="pretty-print an EML tree")
    rn.add_argument("eml_expr")
    rn.add_argument("--format", choices=["text", "rpn", "latex", "tree"], default="tree")
    rn.set_defaults(func=_cmd_render)

    ev = sub.add_parser("eval", help="numerically evaluate an EML tree")
    ev.add_argument("eml_expr")
    ev.add_argument(
        "--bind",
        dest="binding",
        action="append",
        default=[],
        help="variable binding NAME=VALUE; repeatable",
    )
    ev.set_defaults(func=_cmd_eval)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
