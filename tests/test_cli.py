import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(ROOT / "src")}
    # inherit existing env
    import os

    env = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, "-m", "eml.cli", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )


def test_cli_translate_exp():
    r = _run(["translate", "exp(x)"])
    assert r.returncode == 0
    assert "eml(x, 1)" in r.stdout


def test_cli_reverse():
    r = _run(["reverse", "eml(x, 1)"])
    assert r.returncode == 0
    assert "exp(x)" in r.stdout


def test_cli_verify_true():
    r = _run(["verify", "eml(x, 1)", "--as", "exp(x)"])
    assert r.returncode == 0
    assert "match  : True" in r.stdout


def test_cli_identities_lists_known():
    r = _run(["identities"])
    assert r.returncode == 0
    assert "exp(x)" in r.stdout
    assert "ln(x)" in r.stdout


def test_cli_eval():
    r = _run(["eval", "eml(x, 1)", "--bind", "x=1.0"])
    assert r.returncode == 0
    # exp(1) ~ 2.718
    assert "2.71" in r.stdout


def test_cli_rpn_format():
    r = _run(["translate", "exp(x)", "--format", "rpn"])
    assert r.returncode == 0
    assert r.stdout.strip() == "x 1 eml"
