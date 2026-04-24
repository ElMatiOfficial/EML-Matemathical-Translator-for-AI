"""Rendering EML trees in several formats.

Supported output forms:

* :func:`to_text`  -- infix-ish textual form: ``eml(1, eml(x, 1))``
* :func:`to_rpn`   -- reverse-Polish, matching the K-length used in the paper
* :func:`to_latex` -- LaTeX form suitable for papers or Jupyter display
* :func:`to_tree`  -- indented ASCII tree, useful for debugging
"""

from __future__ import annotations

from eml.ast import Eml, Node, One, Var


def to_text(node: Node) -> str:
    """Pretty textual form: ``eml(1, eml(x, 1))``."""
    if isinstance(node, One):
        return "1"
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Eml):
        return f"eml({to_text(node.left)}, {to_text(node.right)})"
    raise TypeError(f"Unknown node type: {type(node).__name__}")


def to_rpn(node: Node, sep: str = " ") -> str:
    """Reverse-Polish notation, matching the paper's K-length metric.

    Example: ``exp(x) = eml(x, 1)`` becomes ``"x 1 eml"`` (K=3).
    """
    tokens: list[str] = []
    _collect_rpn(node, tokens)
    return sep.join(tokens)


def _collect_rpn(node: Node, out: list[str]) -> None:
    if isinstance(node, One):
        out.append("1")
        return
    if isinstance(node, Var):
        out.append(node.name)
        return
    if isinstance(node, Eml):
        _collect_rpn(node.left, out)
        _collect_rpn(node.right, out)
        out.append("eml")
        return
    raise TypeError(f"Unknown node type: {type(node).__name__}")


def to_latex(node: Node) -> str:
    r"""LaTeX form using the ``\operatorname{eml}`` macro.

    Example: ``\operatorname{eml}(1,\, \operatorname{eml}(x,\, 1))``.
    """
    if isinstance(node, One):
        return "1"
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Eml):
        return (
            r"\operatorname{eml}("
            + to_latex(node.left)
            + r",\, "
            + to_latex(node.right)
            + ")"
        )
    raise TypeError(f"Unknown node type: {type(node).__name__}")


def to_tree(node: Node, indent: str = "  ") -> str:
    """Indented tree form, one node per line.

    Uses pure-ASCII indentation so the output is safe on every terminal,
    including the default Windows console (cp1252).

    Example::

        eml
          L: 1
          R: eml
            L: x
            R: 1
    """
    lines: list[str] = []
    _collect_tree(node, 0, None, lines, indent)
    return "\n".join(lines)


def _collect_tree(
    node: Node, depth: int, role: str | None, lines: list[str], indent: str
) -> None:
    pad = indent * depth
    tag = f"{role}: " if role else ""
    if isinstance(node, One):
        lines.append(f"{pad}{tag}1")
        return
    if isinstance(node, Var):
        lines.append(f"{pad}{tag}{node.name}")
        return
    if isinstance(node, Eml):
        lines.append(f"{pad}{tag}eml")
        _collect_tree(node.left, depth + 1, "L", lines, indent)
        _collect_tree(node.right, depth + 1, "R", lines, indent)
        return
    raise TypeError(f"Unknown node type: {type(node).__name__}")


__all__ = ["to_text", "to_rpn", "to_latex", "to_tree"]
