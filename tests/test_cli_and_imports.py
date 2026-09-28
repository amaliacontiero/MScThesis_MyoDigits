"""Import every package module, and smoke-test every command-line entry point."""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO / "src")}

MODULES = sorted(
    ".".join(p.relative_to(REPO / "src").with_suffix("").parts)
    for p in (REPO / "src").rglob("*.py") if p.name != "__init__.py"
)


@pytest.mark.parametrize("module", MODULES)
def test_package_module_imports(module):
    importlib.import_module(module)


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=ENV,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("script", ["scripts/analysis/extract_r2_statistics.py",
                                    "scripts/release/import_run.py",
                                    "scripts/analysis/strict_tx_sensitivity.py",
                                    "scripts/analysis/r2_reconciliation.py",
                                    "scripts/release/generate_tables.py"])
def test_cli_help(script):
    r = run(script, "--help")
    assert r.returncode == 0 and "usage" in r.stdout.lower(), r.stderr


def test_every_script_compiles():
    for p in list((REPO / "scripts").rglob("*.py")) + list((REPO / "figures").rglob("*.py")) + [REPO / "launch.py"]:
        compile(p.read_text(), str(p), "exec")
