#!/usr/bin/env python3
# --- configurable roots -------------------------------------------------
# The analysis and figure scripts read run directories that are not
# distributed with this repository. Point these at your own copies.
import os as _os
from pathlib import Path as _Path
_ROOT   = _Path(_os.environ.get("MYODIGITS_ROOT", _Path(__file__).resolve().parents[2]))
_THESIS = _Path(_os.environ.get("MYODIGITS_THESIS_DIR", _ROOT))
_EGOEMG = _Path(_os.environ.get("MYODIGITS_EGOEMG_DIR", _ROOT.parent / "MD_EgoEMG"))
# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""Per-joint coefficient of determination for the headline model.

Each joint is scored against its own mean, per participant, then averaged over
participants. Also records how often a joint scores below zero, which happens
when the prediction is worse than simply emitting that joint's mean angle.

Writes figures2/perjoint_r2.json.
"""
import glob, json, os
import numpy as np

TH = str(_THESIS)
CACHE = f"{TH}/figures/oracle_v4_cache"
OUT = f"{TH}/figures2/perjoint_r2.json"

J = ["thumb CMC spread", "thumb CMC stretch", "thumb MCP stretch", "thumb IP stretch",
     "index MCP spread", "index MCP stretch", "index PIP stretch", "index DIP stretch",
     "middle MCP spread", "middle MCP stretch", "middle PIP stretch", "middle DIP stretch",
     "ring MCP spread", "ring MCP stretch", "ring PIP stretch", "ring DIP stretch",
     "pinky MCP spread", "pinky MCP stretch", "pinky PIP stretch", "pinky DIP stretch"]

R2, MAE, RNG = [], [], []
for p in sorted(glob.glob(f"{CACHE}/U*.npz")):
    z = np.load(p)
    P, T = z["pred"].astype(np.float64), z["targ"].astype(np.float64)
    mu = T.mean(0, keepdims=True)
    sse = ((P - T) ** 2).sum(0)
    sst = ((T - mu) ** 2).sum(0)
    R2.append(1 - sse / sst)
    MAE.append(np.abs(P - T).mean(0))
    RNG.append(np.percentile(T, 98, axis=0) - np.percentile(T, 2, axis=0))
    del P, T, z

R2, MAE, RNG = np.array(R2), np.array(MAE), np.array(RNG)
mean_r2, mean_mae, mean_rng = R2.mean(0), MAE.mean(0), RNG.mean(0)
n_neg = (R2 < 0).sum(0)

order = np.argsort(-mean_r2)
print(f"{'joint':22s} {'R2':>7} {'MAE':>7} {'range':>7} {'neg':>5}")
for j in order:
    print(f"{J[j]:22s} {mean_r2[j]:7.3f} {mean_mae[j]:7.2f} {mean_rng[j]:7.1f} "
          f"{n_neg[j]:3d}/21")

sp = [i for i, n in enumerate(J) if "spread" in n]
st = [i for i, n in enumerate(J) if "stretch" in n]
print(f"\nmean R2 over all joints        {mean_r2.mean():.3f}")
print(f"  spread joints                {mean_r2[sp].mean():.3f}")
print(f"  stretch joints               {mean_r2[st].mean():.3f}")
print(f"joints negative on some fold   {(n_neg > 0).sum()} of 20")
print(f"participants with >=1 negative {(R2 < 0).any(1).sum()} of 21")
print(f"correlation of per-joint R2 with range of motion: "
      f"{np.corrcoef(mean_r2, mean_rng)[0,1]:+.3f}")

json.dump(dict(joints=J,
               r2=mean_r2.tolist(), mae=mean_mae.tolist(), rng=mean_rng.tolist(),
               n_negative=n_neg.tolist(),
               mean_r2=float(mean_r2.mean()),
               spread_r2=float(mean_r2[sp].mean()),
               stretch_r2=float(mean_r2[st].mean()),
               participants_with_negative=int((R2 < 0).any(1).sum()),
               corr_r2_range=float(np.corrcoef(mean_r2, mean_rng)[0, 1])),
          open(OUT, "w"), indent=1)
print("\nwrote", OUT)
