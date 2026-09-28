"""Shared test setup.

The suite never needs a GPU or the private dataset. CUDA is hidden before torch
is imported anywhere, so a test run cannot compete with a training job.
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "normalisation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def repo():
    return REPO
