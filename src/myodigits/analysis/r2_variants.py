"""The three coefficient-of-determination variants, one implementation.

Twenty joints are predicted at once, so R-squared needs an explicit choice of
reference mean. For one held-out participant, with references ``T`` and
predictions ``P`` of shape ``(n_samples, J)``:

    SSE_j = sum_t (T_tj - P_tj)^2
    SST_j = sum_t (T_tj - mean_t T_tj)^2          each joint about its own mean

joint-centred aggregate      1 - sum_j SSE_j / sum_j SST_j
    How much of the total within-joint reference variation is recovered.
    Joints contribute in proportion to how much they move. This is the variant
    reported throughout the thesis (Equation 4.4).

uniform mean per-joint       (1/J) sum_j [1 - SSE_j / SST_j]
    How well a typical joint is recovered. Every joint weighs 1/J however
    little it moves, so small-range joints dominate.

globally pooled              1 - sum_j SSE_j / SST_global
    SST_global is taken about one scalar mean over every joint and sample, so
    between-joint differences in resting angle sit in the denominator. This is
    the value ``eval.py`` stores as ``test_r2``. It is not a performance
    measure: a constant posture that ignores the input scores roughly 0.55.

Each variant is computed within a participant and then averaged uniformly over
folds (``loso_mean``).

All three are exact functions of per-joint sufficient statistics
``(n, SSE_j, SST_j, mean_j)``, because

    SST_global = sum_j SST_j + n * sum_j (mean_j - grand_mean)^2

with ``grand_mean`` the mean of the ``mean_j`` (every joint has ``n`` samples).
The statistics are small enough to publish, so the thesis values can be
recomputed without the prediction arrays or the dataset.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

JOINT_CENTRED = "joint_centred_aggregate"
UNIFORM_MEAN = "uniform_mean_per_joint"
GLOBALLY_POOLED = "globally_pooled"

VARIANTS = (JOINT_CENTRED, UNIFORM_MEAN, GLOBALLY_POOLED)

LABELS = {
    JOINT_CENTRED: "joint-centred aggregate R²",
    UNIFORM_MEAN: "uniform mean per-joint R²",
    GLOBALLY_POOLED: "globally pooled R²",
}


def _as_2d_float64(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D (n_samples, n_joints) array, got shape {a.shape}")
    return a


def r2_variants(pred, targ) -> dict:
    """All three variants directly from arrays, for one participant."""
    P, T = _as_2d_float64(pred), _as_2d_float64(targ)
    if P.shape != T.shape:
        raise ValueError(f"pred {P.shape} and targ {T.shape} differ in shape")
    sse_j = ((T - P) ** 2).sum(axis=0)
    sst_j = ((T - T.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    sst_global = ((T - T.mean()) ** 2).sum()
    return {
        JOINT_CENTRED: float(1.0 - sse_j.sum() / sst_j.sum()),
        UNIFORM_MEAN: float(np.mean(1.0 - sse_j / sst_j)),
        GLOBALLY_POOLED: float(1.0 - sse_j.sum() / sst_global),
    }


def sufficient_statistics(pred, targ, chunk_rows: int = 250_000) -> dict:
    """Per-joint ``n, SSE_j, SST_j, mean_j`` for one participant.

    Accumulated in float64 over row chunks so that large float32 arrays do not
    need a full float64 copy in memory.
    """
    P = np.asarray(pred)
    T = np.asarray(targ)
    if P.shape != T.shape or P.ndim != 2:
        raise ValueError(f"pred {P.shape} and targ {T.shape} must be equal 2-D shapes")
    n, J = T.shape
    if n < 2:
        raise ValueError("need at least two samples")

    sum_t = np.zeros(J)
    sse = np.zeros(J)
    for s in range(0, n, chunk_rows):
        t = T[s:s + chunk_rows].astype(np.float64)
        p = P[s:s + chunk_rows].astype(np.float64)
        sum_t += t.sum(axis=0)
        sse += ((t - p) ** 2).sum(axis=0)
    mean = sum_t / n

    sst = np.zeros(J)
    for s in range(0, n, chunk_rows):
        t = T[s:s + chunk_rows].astype(np.float64)
        sst += ((t - mean) ** 2).sum(axis=0)

    return {
        "n_samples": int(n),
        "n_joints": int(J),
        "sse_per_joint": sse.tolist(),
        "sst_per_joint": sst.tolist(),
        "mean_per_joint": mean.tolist(),
    }


def global_total_sum_of_squares(stats: Mapping) -> float:
    """SST about one grand mean, reconstructed exactly from per-joint statistics."""
    n = int(stats["n_samples"])
    sst_j = np.asarray(stats["sst_per_joint"], dtype=np.float64)
    mean_j = np.asarray(stats["mean_per_joint"], dtype=np.float64)
    return float(sst_j.sum() + n * ((mean_j - mean_j.mean()) ** 2).sum())


def r2_from_statistics(stats: Mapping) -> dict:
    """All three variants from ``sufficient_statistics`` output."""
    sse_j = np.asarray(stats["sse_per_joint"], dtype=np.float64)
    sst_j = np.asarray(stats["sst_per_joint"], dtype=np.float64)
    if sse_j.shape != sst_j.shape:
        raise ValueError("sse_per_joint and sst_per_joint differ in length")
    if np.any(sst_j <= 0):
        raise ValueError("a joint has zero reference variance; per-joint R² is undefined")
    return {
        JOINT_CENTRED: float(1.0 - sse_j.sum() / sst_j.sum()),
        UNIFORM_MEAN: float(np.mean(1.0 - sse_j / sst_j)),
        GLOBALLY_POOLED: float(1.0 - sse_j.sum() / global_total_sum_of_squares(stats)),
    }


def rmse_discrepancy(stats: Mapping, test_rmse: float) -> float:
    """Absolute difference between a stored RMSE and the RMSE implied by SSE.

    Used to confirm that published statistics belong to the stored fold record.
    """
    sse = float(np.sum(stats["sse_per_joint"]))
    n_values = int(stats["n_samples"]) * int(stats["n_joints"])
    return abs(float(np.sqrt(sse / n_values)) - float(test_rmse))


def loso_mean(per_fold: Iterable[Mapping], variant: str) -> float:
    """Uniform mean of one variant over folds (each fold weighted equally)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    values = [float(f[variant]) for f in per_fold]
    if not values:
        raise ValueError("no folds")
    return float(np.mean(values))
