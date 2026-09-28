"""Paired, participant-level statistics used for every between-run comparison.

The held-out participant is the experimental unit. Two runs are compared on the
same participants, so tests and intervals operate on per-participant values.

These quantify consistency across held-out participants *for the specific
trained runs being compared*. No training seed was fixed or repeated in this
project, so they do not include uncertainty from retraining a model.
"""
from __future__ import annotations

import numpy as np
import scipy
from scipy import stats

WILCOXON_PARAMETERS = {
    "alternative": "two-sided",
    "zero_method": "wilcox",
    "correction": False,
    "method": "exact",
}
BOOTSTRAP_SEED = 20260910
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95


def paired_wilcoxon(a, b) -> dict:
    """Wilcoxon signed-rank test on ``a - b`` with every parameter pinned."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    res = stats.wilcoxon(a, b, **WILCOXON_PARAMETERS)
    return {
        "W": float(res.statistic),
        "p": float(res.pvalue),
        "n_pairs": int(len(a)),
        "parameters": {"function": "scipy.stats.wilcoxon", **WILCOXON_PARAMETERS,
                       "scipy_version": scipy.__version__},
    }


def bootstrap_mean_ci(differences, seed: int = BOOTSTRAP_SEED,
                      n_resamples: int = N_BOOTSTRAP, level: float = CI_LEVEL) -> dict:
    """Percentile bootstrap interval for the mean paired difference.

    Participants are resampled with replacement. The generator call pattern is
    fixed (``default_rng(seed).integers(0, n, size=(n_resamples, n))``) so the
    published interval is reproducible exactly.
    """
    d = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    boot = d[idx].mean(axis=1)
    tail = 100.0 * (1.0 - level) / 2.0
    return {
        "low": float(np.percentile(boot, tail)),
        "high": float(np.percentile(boot, 100.0 - tail)),
        "level": level,
        "method": "percentile bootstrap, participants resampled with replacement, "
                  "statistic = mean paired difference",
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "rng": f"numpy.random.default_rng (numpy {np.__version__})",
    }


def associations(a, b) -> dict:
    """Pearson and Spearman association of per-participant values across two runs."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pr = stats.pearsonr(a, b)
    sr = stats.spearmanr(a, b)
    return {
        "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
        "spearman_rho": float(sr[0]), "spearman_p": float(sr[1]),
    }


def describe(values) -> dict:
    """Mean, both standard deviation conventions, extremes and worst/best ratio."""
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "sd_ddof0": float(v.std(ddof=0)),
        "sd_ddof1": float(v.std(ddof=1)),
        "best": float(v.min()),
        "worst": float(v.max()),
        "worst_over_best_ratio": float(v.max() / v.min()),
    }
