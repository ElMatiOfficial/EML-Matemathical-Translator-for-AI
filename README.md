# EML Translator

A Python library that translates between standard mathematical expressions and
**EML** (Exp-Minus-Log) trees — the universal reduction primitive for
elementary mathematics introduced by Odrzywołek (2026).

> Odrzywołek showed that a single binary operator,
>
> $$\mathrm{eml}(x, y) \;=\; \exp(x) - \ln(y),$$
>
> combined with the single literal constant `1`, is sufficient to express
> every elementary function. It is the continuous-mathematics analogue of
> the NAND gate in digital logic.
>
> — *[All elementary functions from a single operator](https://arxiv.org/abs/2603.21852)*, arXiv:2603.21852.

This library is built for mathematical AI researchers who want to
**feed math to a model in a single-primitive normal form**, **parse
EML outputs back into human math**, and **discover new EML identities**
by exhaustive symbolic search.

---

## Why EML for AI?

An LLM or a symbolic-regression model has to choose, at every node, among a
large menu of operators (`+`, `-`, `*`, `/`, `^`, `sin`, `cos`, `log`, ...).
EML collapses that menu to a single gate. The model only ever has to decide
between the literal `1`, a variable, or `eml(·, ·)`. This:

- shrinks the search space in symbolic regression,
- gives a uniform token vocabulary for training,
- makes tree edit distance a meaningful similarity metric,
- and exposes the *Kolmogorov length* (RPN token count) of each identity
  as an intrinsic complexity score.

---

## Install

```bash
pip install -e .
```

Requires Python 3.9+ and [`sympy`](https://www.sympy.org/).

---

## Quickstart

### Python API

```python
import eml

# math -> EML
tree = eml.to_eml("exp(x) - 1")
print(eml.to_text(tree))
# -> eml(eml(1, eml(eml(1, eml(x, 1)), 1)), eml(1, 1))

# EML -> math
tree = eml.parse("eml(x, 1)")
print(eml.from_eml(tree))
# -> exp(x)

# Numeric evaluation
print(eml.evaluate(eml.to_eml("exp(x)"), {"x": 2.0}))
# -> 7.38905609893065

# Kolmogorov length (RPN token count) — matches the paper
print(eml.to_rpn(eml.to_eml("exp(x)")))  # "x 1 eml"  (K = 3)
print(eml.to_rpn(eml.to_eml("log(x)")))  # K = 7
```

### Command line

```bash
# math -> EML
eml translate "exp(x) - 1"

# EML -> math
eml reverse "eml(x, 1)"

# List every registered identity
eml identities

# Verify an EML tree equals a math expression
eml verify "eml(x, 1)" --as "exp(x)"

# Numeric evaluation
eml eval "eml(x, 1)" --bind x=2.0

# Exhaustive search for a new EML decomposition
eml search "exp(x)" --vars x --max-size 5

# Render a tree in several forms
eml translate "log(x)" --format rpn     # 1 1 x eml 1 eml eml
eml translate "log(x)" --format latex   # \operatorname{eml}(1,\, ...)
eml render "eml(1, eml(eml(1, x), 1))" --format tree
```

---

## What's in the box

### Verified identities

Every builtin identity is numerically verified at import time against a
reference implementation. The three identities explicitly stated in the
main paper are marked with the citation; the rest are ported from the
paper's reference implementation
([SymbolicRegressionPackage](https://github.com/VA00/SymbolicRegressionPackage),
`EML_toolkit/EmL_compiler/eml_compiler_v4.py`) and re-verified here.

| Name       | K (RPN length) | Source |
|------------|----------------|--------|
| `exp(x)`   | 3              | Odrzywołek (2026), main paper |
| `ln(x)`    | 7              | Odrzywołek (2026), main paper |
| `e`        | 3              | Odrzywołek (2026), main paper |
| `0`        | 7              | derived (= ln 1) |
| `x - y`    | 11             | derived |
| `1 - y`, `x - 1` | 11       | derived (compact specialisations) |
| `-x`       | 17             | SymbolicRegressionPackage |
| `x + y`    | 27             | SymbolicRegressionPackage |
| `1/x`      | 25             | SymbolicRegressionPackage |
| `x * y`    | **41**         | SymbolicRegressionPackage (matches paper's stated K for multiplication) |
| `x / y`    | 65             | SymbolicRegressionPackage |
| `x ^ y`    | 49             | SymbolicRegressionPackage |
| `sqrt(x)`  | 99             | derived from `pow` |
| `2`        | 27             | SymbolicRegressionPackage |
| `I`        | 115            | SymbolicRegressionPackage (sign-corrected, see identities.py) |
| `π`        | **193**        | SymbolicRegressionPackage (matches paper's stated K for π) |

All trigonometric and hyperbolic functions (`sin`, `cos`, `tan`, `sinh`,
`cosh`, `tanh`, and their inverses) translate via `sympy`'s `rewrite(exp)`
chain, landing in the same `exp`/`log`/`pow`/`I` primitives above. No
separate identity registration is required — the forward compiler handles
them automatically.

You can register additional identities — see
[**Extending the library**](#extending-the-library) below.

### Modules

| Module               | Purpose                                                     |
|----------------------|-------------------------------------------------------------|
| `eml.ast`            | `One`, `Var`, `Eml` nodes; size/depth/substitution helpers  |
| `eml.eval`           | Numeric evaluation over real and complex domains            |
| `eml.identities`     | Registry of verified EML decompositions                     |
| `eml.forward`        | Standard math (via `sympy`) → EML tree                      |
| `eml.inverse`        | EML tree → `sympy` expression                               |
| `eml.parser`         | Parse textual and RPN forms                                 |
| `eml.printer`        | Render as text, RPN, LaTeX, or indented tree                |
| `eml.search`         | Exhaustive search for new EML identities                    |
| `eml.cli`            | `eml` command-line tool                                     |

---

## Extending the library

The identities above cover the full paper chain (exp, log, arithmetic,
powers, trig via rewrite). For new primitives — special functions, custom
activations, piecewise definitions — this library gives you two
complementary ways to add them.

### 1. Register a closed-form decomposition

If you already know the EML tree (from the paper's supplementary, from a
derivation on paper, or from your own search), register it:

```python
import cmath
from eml import Eml, One, Var, register_identity

# Hypothetical identity: myfunc(x) = eml(eml(x, 1), 1)
def myfunc(x):
    return Eml(Eml(x, One()), One())

# verify against a reference numeric implementation (required by default)
register_identity("myfunc", myfunc, reference=lambda x: cmath.exp(cmath.exp(x)))
```

Registration refuses to install an identity that disagrees with its
reference at any sample point, so you cannot accidentally pollute the
registry with bad decompositions.

### 2. Search for one

The `search_identity` function enumerates all EML trees up to a size
budget, evaluates each one on a sample grid, and returns the trees that
match a target function numerically. This is the same idea the paper uses
to discover candidate identities.

```python
import cmath
from eml import search_identity, to_text, tree_size

matches = search_identity(cmath.exp, variables=["x"], max_size=5)
for t in matches:
    print(tree_size(t), to_text(t))
# -> 3 eml(x, 1)
```

Search is exponential in `max_size`; practical limits are around 9–11 for a
single variable on a modern laptop. Identities of high Kolmogorov length
(e.g. multiplication at K=41, π at K=193) are out of brute-force reach and
require the heuristic methods described in the paper.

---

## Numeric domain

The evaluator uses the principal branch of the complex logarithm, matching
the paper. Real-valued inputs on the positive axis take the fast real path;
everything else automatically falls through to `cmath`. Endpoints of the
domain (e.g. `ln(0) = -∞`) are not representable over the ground constant
`1` alone and are skipped during verification rather than silently producing
garbage.

---

## Citing

If this library supports your research, please cite both the library and
the original paper:

```bibtex
@article{Odrzywolek2026EML,
  title   = {All elementary functions from a single operator},
  author  = {Odrzywo{\l}ek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.21852},
}

@software{eml_translator,
  title  = {EML Translator: A library for translating between standard mathematics and EML},
  author = {ElMatiOfficial and contributors},
  url    = {https://github.com/ElMatiOfficial/EML-Matemathical-Translator-for-AI},
  year   = {2026},
}
```

A [CITATION.cff](CITATION.cff) file is also included for automatic tooling.

---

## License

[MIT](LICENSE) — free for research, teaching, and commercial use.

## Contributing

Issues and pull requests are welcome. The most valuable contributions right
now are:

- **New verified identities** — especially the paper's supplementary
  decompositions for `+`, `*`, `/`, `sin`, `cos`, `sqrt`, etc.
- **Faster search heuristics** — the current exhaustive enumerator works
  but does not yet prune based on structural symmetries.
- **Alternative EML variants** — the paper also identifies `edl(x, y) =
  exp(x)/ln(y)` and `ln(x) - exp(y)` as sufficient operators. A second
  backend for those would be a great addition.
