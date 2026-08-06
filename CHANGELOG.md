# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Ruff lint + format checks in CI, with configuration in `pyproject.toml`.
- Dependabot for GitHub Actions and pip, monthly and grouped.
- PEP 561 `py.typed` marker, so type hints are visible to installed consumers.
- `CHANGELOG.md` (this file).

### Changed
- License metadata moved to the PEP 639 SPDX form (`license = "MIT"` plus
  `license-files`), replacing the deprecated `{ file = "LICENSE" }` table and
  the `License :: OSI Approved ::` classifier. setuptools >= 77 errors when
  both forms are present.
- CI matrix extended to Python 3.13; the build job now runs
  `twine check --strict` and installs the built wheel into a clean virtualenv
  before smoke-testing the `eml` CLI.
- The publish workflow fires on a published GitHub Release rather than a raw
  tag push, and fails if the tag disagrees with `version` in `pyproject.toml`.

## [0.1.0] — unreleased

First release. Not yet published to PyPI.

### Added
- `eml` package: AST, parser, printer, forward and inverse translation between
  standard mathematical expressions and EML (Exp-Minus-Log) trees.
- Numeric evaluation with complex-branch handling.
- Identity registry ported from the supplementary material of
  [arXiv:2603.21852](https://arxiv.org/abs/2603.21852), plus exhaustive
  identity search.
- `eml` command-line interface.
