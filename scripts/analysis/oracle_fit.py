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
"""Fit the oracle calibration ladder on cached v4 predictions.
Oracles are FITTED on a subsample (parameters are 2 per joint, so a subsample is
ample) and EVALUATED on all windows. This is statistically equivalent to fitting
on everything and ~20x faster."""
import os, glob, json
import numpy as np
from sklearn.isotonic import IsotonicRegression

CACHE = str(_THESIS / "figures/oracle_v4_cache")
OUT   = str(_THESIS / "figures/oracle_v4.json")
FIT_N = 200_000          # subsample used to FIT oracle parameters

def mae(a, b): return float(np.abs(a - b).mean())

def affine_ls(P, T, idx):
    O = np.empty_like(P)
    for j in range(P.shape[1]):
        x, y = P[idx, j], T[idx, j]
        A = np.vstack([x, np.ones_like(x)]).T
        g, b = np.linalg.lstsq(A, y, rcond=None)[0]
        O[:, j] = g * P[:, j] + b
    return O

def affine_mae(P, T, idx, grid=41):
    O = np.empty_like(P)
    for j in range(P.shape[1]):
        x, y = P[idx, j], T[idx, j]
        sx = x.std() or 1.0
        gains = np.linspace(0.2, 3.0, grid) * (y.std() / sx)
        best, bg, bb = None, 1.0, 0.0
        for g in gains:
            b = np.median(y - g * x)                 # on the subsample only
            e = np.abs(g * x + b - y).mean()
            if best is None or e < best: best, bg, bb = e, g, b
        O[:, j] = bg * P[:, j] + bb
    return O

def isotonic(P, T, idx):
    O = np.empty_like(P)
    for j in range(P.shape[1]):
        ir = IsotonicRegression(out_of_bounds="clip").fit(P[idx, j], T[idx, j])
        O[:, j] = ir.predict(P[:, j])
    return O

rows = {}
files = sorted(glob.glob(f"{CACHE}/U*.npz"))
print(f"fitting oracles on {len(files)} subjects (fit subsample = {FIT_N:,})")
for f in files:
    u = int(os.path.basename(f)[1:3])
    z = np.load(f); P = z["pred"].astype(np.float64); T = z["targ"].astype(np.float64)
    rs = np.random.RandomState(0)
    idx = rs.choice(len(P), min(FIT_N, len(P)), replace=False)
    rows[u] = {"baseline":   mae(P, T),
               "affine_ls":  mae(affine_ls(P, T, idx), T),
               "affine_mae": mae(affine_mae(P, T, idx), T),
               "isotonic":   mae(isotonic(P, T, idx), T)}
    print(f"  U{u:02d}  " + "  ".join(f"{k}={v:.3f}" for k, v in rows[u].items()), flush=True)

res = {}
for k in ["baseline", "affine_ls", "affine_mae", "isotonic"]:
    v = np.array([rows[u][k] for u in sorted(rows)])
    res[k] = {"mean": float(v.mean()), "std": float(v.std()), "best": float(v.min()),
              "worst": float(v.max()), "spread": float(v.max()/v.min())}
res["per_user"] = {str(u): rows[u] for u in sorted(rows)}
res["fit_subsample"] = FIT_N
json.dump(res, open(OUT, "w"), indent=2)

print("\n=== ORACLE CALIBRATION CEILING, CORRECTED v4 (run28) ===")
print("%-13s%9s%8s%8s%9s" % ("condition", "mean", "best", "worst", "spread"))
for k in ["baseline", "affine_ls", "affine_mae", "isotonic"]:
    r = res[k]; print("%-13s%9.3f%8.3f%8.3f%8.2fx" % (k, r["mean"], r["best"], r["worst"], r["spread"]))
b, i = res["baseline"], res["isotonic"]
print("\nspread %.2fx -> %.2fx; closes %.1f%% of the excess above 1.0"
      % (b["spread"], i["spread"], 100*(b["spread"]-i["spread"])/(b["spread"]-1.0)))
