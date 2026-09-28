"""Experiment A: strict normalisation protocol.

The definitive runs standardise every participant with that participant's own
full-record statistics, including the held-out one. This module implements the
stricter alternative: statistics are fitted on the 17 TRAINING participants of
the fold only, and applied unchanged to training, validation and test.

Exactness. The stored .npz windows already hold x_norm = (x_raw - mu_u)/sd_u
with mu_u, sd_u recorded per participant in meta.json. The raw value is therefore
recoverable exactly (to float32), so no re-preprocessing is required and the
filtering, RMS, windowing and fold assignment stay bit-identical to the original:

    x_strict = (x_norm * sd_u + mu_u - mu_T) / sd_T

Pooling is PARTICIPANT-WEIGHTED, not sample-weighted. Justification: the thesis
states throughout that the independent unit is the participant, not the window;
participants contribute near-identical window counts (37,884 each), so the two
schemes almost coincide; and equal weighting stops a marginally longer recording
from dominating a population statistic. The choice is recorded in the fold
artefact so it is auditable.
"""
import json
from pathlib import Path

import numpy as np
import torch

EPS = 1e-8


def load_meta(data_root):
    return json.load(open(Path(data_root) / "meta.json"))


def _user_key(user_dir):
    return Path(user_dir).name                      # "User7"


def fold_statistics(meta, train_users, *, exclude_users):
    """Participant-weighted pooled statistics over the training participants only.

    exclude_users must contain the test and validation participants. The
    assertion below is the guarantee that they cannot enter the statistics.
    """
    pu = meta["per_user"]
    train_keys = [f"User{u}" for u in train_users]

    # ---- hard guarantees -------------------------------------------------
    excl = {f"User{u}" for u in exclude_users}
    leaked = excl.intersection(train_keys)
    assert not leaked, f"LEAK: excluded participants present in fitting set: {sorted(leaked)}"
    assert len(train_keys) == len(set(train_keys)), "duplicate training participants"
    for k in train_keys:
        assert k in pu, f"missing statistics for {k}"

    def pool(mean_key, std_key, dim):
        M = np.array([pu[k][mean_key] for k in train_keys], dtype=np.float64)
        S = np.array([pu[k][std_key] for k in train_keys], dtype=np.float64)
        assert M.shape == (len(train_keys), dim), M.shape
        mu = M.mean(axis=0)                                   # participant-weighted
        second = (S ** 2 + M ** 2).mean(axis=0)               # E[x^2] under equal weights
        var = np.maximum(second - mu ** 2, 0.0)
        sd = np.sqrt(var)
        # deterministic handling of degenerate scale
        degenerate = sd < EPS
        sd = np.where(degenerate, 1.0, sd)
        return mu, sd, int(degenerate.sum())

    emg_mu, emg_sd, emg_deg = pool("emg_mean", "emg_std", 16)
    jnt_mu, jnt_sd, jnt_deg = pool("jnt_mean", "jnt_std", 20)

    n_windows = int(sum(pu[k]["windows"] for k in train_keys))
    return {
        "train_users": sorted(train_users),
        "excluded_users": sorted(exclude_users),
        "n_train_participants": len(train_keys),
        "n_contributing_windows": n_windows,
        "weighting": "participant-weighted",
        "emg_mean": emg_mu.tolist(), "emg_std": emg_sd.tolist(),
        "jnt_mean": jnt_mu.tolist(), "jnt_std": jnt_sd.tolist(),
        "degenerate_emg_channels": emg_deg,
        "degenerate_jnt_channels": jnt_deg,
    }


def restandardise_user(X, y, user_key, meta, stats):
    """Undo that participant's own standardisation, apply the fold statistics."""
    pu = meta["per_user"][user_key]
    mu_u = np.asarray(pu["emg_mean"], np.float64)          # (16,)
    sd_u = np.asarray(pu["emg_std"],  np.float64)
    jm_u = np.asarray(pu["jnt_mean"], np.float64)          # (20,)
    js_u = np.asarray(pu["jnt_std"],  np.float64)

    mu_T = np.asarray(stats["emg_mean"], np.float64)
    sd_T = np.asarray(stats["emg_std"],  np.float64)
    jm_T = np.asarray(stats["jnt_mean"], np.float64)
    js_T = np.asarray(stats["jnt_std"],  np.float64)

    # X is (N, 16, 400): channel axis is 1
    Xr = X.astype(np.float64) * sd_u[None, :, None] + mu_u[None, :, None]
    Xs = (Xr - mu_T[None, :, None]) / sd_T[None, :, None]
    # y is (N, 20)
    yr = y.astype(np.float64) * js_u[None, :] + jm_u[None, :]
    ys = (yr - jm_T[None, :]) / js_T[None, :]
    return Xs.astype(np.float32), ys.astype(np.float32)


def denorm_targets(arr, stats):
    """Return predictions/targets to degrees using TRAINING-participant statistics."""
    jm = np.asarray(stats["jnt_mean"], np.float32)
    js = np.asarray(stats["jnt_std"],  np.float32)
    return arr * js + jm


def save_fold_artifact(path, stats):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(path, "w"), indent=2)
