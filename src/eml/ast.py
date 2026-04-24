"""AST for EML expression trees.

The formal grammar (Odrzywolek 2026):

    S  ::= 1 | Var | eml(S, S)

Every node is one of:
  - `One`   -- the atomic constant 1 (the only literal in pure EML)
  - `Var`   -- a named variable placeholder (e.g. Var("x"))
  - `Eml`   -- an internal node with two children, representing eml(L, R)

Numeric literals other than 1 (e.g. e, pi, 2, -1, 0) are NOT leaves;
they must be expressed as nested EML trees over 1 (see `eml.identities`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class One:
    """The atomic constant 1. The only literal in pure EML."""

    def __repr__(self) -> str:
        return "1"


@dataclass(frozen=True)
class Var:
    """A named variable placeholder."""

    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Eml:
    """An internal EML node: eml(left, right) = exp(left) - ln(right)."""

    left: "Node"
    right: "Node"

    def __repr__(self) -> str:
        return f"eml({self.left!r}, {self.right!r})"


Node = Union[One, Var, Eml]


def tree_size(node: Node) -> int:
    """Number of nodes in the tree (leaves + internal eml nodes).

    Corresponds to the 'leaf+op' count used as a complexity metric in the paper.
    """
    if isinstance(node, (One, Var)):
        return 1
    if isinstance(node, Eml):
        return 1 + tree_size(node.left) + tree_size(node.right)
    raise TypeError(f"Unknown node type: {type(node).__name__}")


def tree_depth(node: Node) -> int:
    """Maximum nesting depth of eml operators along any root-to-leaf path."""
    if isinstance(node, (One, Var)):
        return 0
    if isinstance(node, Eml):
        return 1 + max(tree_depth(node.left), tree_depth(node.right))
    raise TypeError(f"Unknown node type: {type(node).__name__}")


def free_vars(node: Node) -> set[str]:
    """Names of all variables that appear in the tree."""
    if isinstance(node, One):
        return set()
    if isinstance(node, Var):
        return {node.name}
    if isinstance(node, Eml):
        return free_vars(node.left) | free_vars(node.right)
    raise TypeError(f"Unknown node type: {type(node).__name__}")


def substitute(node: Node, mapping: dict[str, Node]) -> Node:
    """Return a new tree with each Var(name) replaced by mapping[name], if present."""
    if isinstance(node, One):
        return node
    if isinstance(node, Var):
        return mapping.get(node.name, node)
    if isinstance(node, Eml):
        return Eml(substitute(node.left, mapping), substitute(node.right, mapping))
    raise TypeError(f"Unknown node type: {type(node).__name__}")
