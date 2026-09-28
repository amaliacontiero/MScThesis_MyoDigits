"""Sequence-position weighting of the evaluation, one implementation.

The definitive evaluation scores 64-window sequences taken at stride 1 inside a
trial, so a window is scored once for every sequence that contains it and once
for every position it occupies inside that sequence. For a trial of ``n``
windows and sequence length ``S`` the valid starts are ``0 .. n-S``, and window
``w`` is scored

    mult(w) = min(w, n-S) - max(0, w-(S-1)) + 1

times. The multiplicities of a trial sum to ``(n-S+1)*S``. A trial shorter than
``S`` contributes no sequences and therefore weight zero.

Why this matters. Any quantity computed over the *unique* windows of a
participant is on a different weighting from the reported metrics, because
interior windows are scored up to ``S`` times and the first and last ``S-1``
fewer. A baseline that is to be compared with a model must be scored with these
weights, or it is not being evaluated on the same thing.

For PiMForce every participant has 66 trials of 574 windows, so the
multiplicities sum to 66 * (574-64+1) * 64 = 2,158,464, which is exactly the
``n_test_windows`` recorded by every definitive run.
"""
from __future__ import annotations

import numpy as np

SEQ_LEN = 64


def window_multiplicity(n_windows: int, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Multiplicity of every window of one trial, as ``uint16``."""
    n, s = int(n_windows), int(seq_len)
    if n < s:
        return np.zeros(max(n, 0), dtype=np.uint16)
    w = np.arange(n)
    return (np.minimum(w, n - s) - np.maximum(0, w - (s - 1)) + 1).astype(np.uint16)


def expected_total(n_windows: int, seq_len: int = SEQ_LEN) -> int:
    """Sequence positions contributed by one trial: ``(n-S+1)*S``, or 0 if too short."""
    n, s = int(n_windows), int(seq_len)
    return 0 if n < s else (n - s + 1) * s


def check_trial(n_windows: int, seq_len: int = SEQ_LEN) -> None:
    """Raise if the multiplicities do not sum to the expected count."""
    got = int(window_multiplicity(n_windows, seq_len).sum())
    want = expected_total(n_windows, seq_len)
    if got != want:
        raise ValueError(f"multiplicity sum {got} != expected {want} for n={n_windows}")


def weighted_mae(y, posture, weights) -> float:
    """Mean absolute error of a constant posture, weighted by sequence position.

    ``y`` is ``(n_windows, J)`` in degrees, ``posture`` is ``(J,)`` in degrees and
    ``weights`` is ``(n_windows,)``. The mean is over weighted rows and joints,
    matching ``eval.py``'s flat mean over every scored element.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    err = np.abs(y - np.asarray(posture, dtype=np.float64)[None, :])
    return float((w[:, None] * err).sum() / (w.sum() * y.shape[1]))
