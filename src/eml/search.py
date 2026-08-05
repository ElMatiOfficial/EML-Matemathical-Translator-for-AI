"""Exhaustive search for EML decompositions of target functions.

This mirrors the candidate-identity discovery procedure used in
Odrzywolek (2026): enumerate EML trees in order of increasing size
(RPN length / K-length), evaluate each on a numeric sample grid, and
retain the ones that match the target function to within a tolerance.

Because the search space grows roughly as the Catalan numbers times the
number of leaf choices, this is only practical for small trees (sizes up
to ~12 with 2 variables). Larger identities (multiplication's K=41, pi's
K=193) are beyond brute-force reach and require the heuristics described
in the paper -- which this module exposes as hooks rather than implements.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from typing import Callable

from eml.ast import Eml, Node, One, Var
from eml.eval import evaluate


def _enumerate_trees(leaves: list[Node], max_size: int) -> Iterator[tuple[int, list[Node]]]:
    """Yield (size, trees-of-that-size) in increasing size order.

    ``size`` is the total node count (leaves + internal eml nodes),
    matching :func:`eml.ast.tree_size`.
    """
    by_size: dict[int, list[Node]] = {1: list(leaves)}
    yield 1, by_size[1]
    for n in range(3, max_size + 1, 2):
        # internal eml-tree sizes are always odd (leaf=1, eml(L,R) adds 1 to L+R)
        level: list[Node] = []
        for l_size in range(1, n - 1, 2):
            r_size = n - 1 - l_size
            if r_size < 1 or r_size % 2 == 0:
                continue
            for left in by_size.get(l_size, ()):
                for right in by_size.get(r_size, ()):
                    level.append(Eml(left, right))
        by_size[n] = level
        yield n, level


def _fingerprint(
    tree: Node, sample_grid: Sequence[dict[str, complex]]
) -> tuple[complex, ...] | None:
    """Evaluate ``tree`` on every sample binding and return the value tuple.

    Returns ``None`` if evaluation overflows or hits a singularity on any
    sample point -- such trees are unhelpful for matching a smooth target.
    """
    out: list[complex] = []
    for binding in sample_grid:
        try:
            v = complex(evaluate(tree, binding))
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        if math.isnan(v.real) or math.isnan(v.imag) or math.isinf(v.real) or math.isinf(v.imag):
            return None
        out.append(v)
    return tuple(out)


def _default_samples(variables: Sequence[str]) -> list[dict[str, complex]]:
    """A small grid of positive-real and complex sample points for each variable."""
    reals = [0.7, 1.3, 2.1, 3.4]
    complex_pts = [complex(1.1, 0.5), complex(0.8, -0.3)]
    base = reals + complex_pts
    grid: list[dict[str, complex]] = []
    for combo in itertools.product(base, repeat=len(variables)):
        grid.append(dict(zip(variables, combo)))
    # keep the grid small -- enough to distinguish but not blow up runtime
    return grid[:24]


def search_identity(
    target: Callable[..., complex],
    variables: Sequence[str] = ("x",),
    *,
    max_size: int = 7,
    tol: float = 1e-8,
    samples: list[dict[str, complex]] | None = None,
    max_results: int = 5,
    verbose: bool = False,
) -> list[Node]:
    """Search for EML trees that equal ``target`` on a numeric sample grid.

    Parameters
    ----------
    target : callable
        Reference numeric implementation. Called as ``target(*values)``
        where ``values`` correspond one-to-one with ``variables``.
    variables : sequence of str
        Names of the free variables to search over. The returned trees
        use :class:`Var` placeholders with these names.
    max_size : int
        Maximum tree size (total node count) to consider. Search is
        exponential in this parameter; practical limit ~11.
    tol : float
        Tolerance for declaring a match on the sample grid.
    samples : list of dicts, optional
        Explicit variable bindings to evaluate on. Each entry must map
        every name in ``variables`` to a number. If omitted, a default
        grid of positive reals and complex points is used.
    max_results : int
        Stop after this many matching trees have been found.
    verbose : bool
        Print progress to stderr as each size level completes.

    Returns
    -------
    list of Node
        Matching trees, ordered by tree size (smallest first).
    """
    sample_grid = samples or _default_samples(variables)
    target_values: list[complex] = []
    for binding in sample_grid:
        target_values.append(complex(target(*[binding[v] for v in variables])))

    leaves: list[Node] = [One()] + [Var(v) for v in variables]
    seen_fingerprints: set[tuple[complex, ...]] = set()
    matches: list[Node] = []

    for size, trees in _enumerate_trees(leaves, max_size):
        if verbose:
            import sys

            print(f"  size={size}: {len(trees)} trees", file=sys.stderr)
        for t in trees:
            fp = _fingerprint(t, sample_grid)
            if fp is None:
                continue
            # deduplicate: we only care about trees with distinct value signatures
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)
            # does it match the target?
            diff = max(abs(a - b) for a, b in zip(fp, target_values))
            scale = max(1.0, max(abs(b) for b in target_values))
            if diff <= tol * scale:
                matches.append(t)
                if len(matches) >= max_results:
                    return matches
    return matches


def verify_match(
    tree: Node,
    target: Callable[..., complex],
    variables: Sequence[str],
    *,
    samples: list[dict[str, complex]] | None = None,
    tol: float = 1e-8,
) -> bool:
    """Return True if ``tree`` matches ``target`` numerically on the sample grid.

    A cheap check the caller can run after narrowing a search with fewer
    samples -- rerun on a denser grid to rule out coincidences.
    """
    sample_grid = samples or _default_samples(variables)
    for binding in sample_grid:
        try:
            lhs = complex(evaluate(tree, binding))
            rhs = complex(target(*[binding[v] for v in variables]))
        except (ValueError, ZeroDivisionError, OverflowError):
            return False
        if abs(lhs - rhs) > tol * max(1.0, abs(rhs)):
            return False
    return True


__all__ = ["search_identity", "verify_match"]
