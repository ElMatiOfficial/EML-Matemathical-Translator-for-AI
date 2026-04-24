"""EML Translator: translate between standard mathematics and EML.

EML (Exp-Minus-Log) is a single binary operator:

    eml(x, y) = exp(x) - ln(y)

Odrzywolek (2026) showed that every elementary function can be expressed
as a nested EML tree over the single constant 1. This library translates
between standard mathematical notation and EML trees in both directions,
and provides a symbolic-regression-style search for new identities.

Paper: https://arxiv.org/abs/2603.21852
"""

from eml.ast import Eml, Node, One, Var, tree_depth, tree_size
from eml.eval import evaluate
from eml.forward import to_eml
from eml.identities import (
    IDENTITIES,
    register_identity,
    verify_identity,
)
from eml.inverse import from_eml
from eml.parser import parse, parse_rpn
from eml.printer import to_latex, to_rpn, to_text, to_tree
from eml.search import search_identity

__version__ = "0.1.0"

__all__ = [
    "Eml",
    "IDENTITIES",
    "Node",
    "One",
    "Var",
    "evaluate",
    "from_eml",
    "parse",
    "parse_rpn",
    "register_identity",
    "search_identity",
    "to_eml",
    "to_latex",
    "to_rpn",
    "to_text",
    "to_tree",
    "tree_depth",
    "tree_size",
    "verify_identity",
]
