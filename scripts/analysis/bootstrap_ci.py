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
"""Bootstrap confidence intervals on the spread metric (resampling subjects)."""
import json, os
import numpy as np

E = str(_ROOT / "experiments/")
OUT = str(_THESIS / "figures")
RUNS = {
 "PoseNetTx":       "run28_posenet_tx_lr3e-4_bs64_ep60_loso_v4headline",
 "PoseNetSeq":      "run29_posenet_seq_lr3e-4_bs64_ep60_loso_v4zoo",
 "TCN-v1":          "run30_tcn_v1_lr3e-4_bs64_ep60_loso_v4zoo",
 "Transfer FT":     "run32_posenet_tx_lr3e-4_bs64_ep60_loso_v4xferFT",
 "Transfer frozen": "run33_posenet_tx_lr3e-4_bs64_ep60_loso_v4xferFrozen",
 "N=4":             "run34_posenet_tx_lr3e-4_bs64_ep60_loso_v4sc04",
 "N=8":             "run35_posenet_tx_lr3e-4_bs64_ep60_loso_v4sc08",
}
B = 10000
rng = np.random.RandomState(0)
res = {}
print(f"{'model':<17}{'spread':>8}{'95% CI':>18}{'mean MAE':>11}{'95% CI':>18}")
for name, r in RUNS.items():
    p = f"{E}{r}/loso_summary.json"
    if not os.path.exists(p): continue
    d = json.load(open(p))
    v = np.array([f["test_mae"] for f in d["per_fold"]])
    if len(v) < 21: continue
    sp, mn = [], []
    for _ in range(B):
        s = v[rng.randint(0, len(v), len(v))]
        sp.append(s.max()/s.min()); mn.append(s.mean())
    lo, hi = np.percentile(sp, [2.5, 97.5]); mlo, mhi = np.percentile(mn, [2.5, 97.5])
    res[name] = {"spread": float(v.max()/v.min()), "spread_ci": [float(lo), float(hi)],
                 "mean": float(v.mean()), "mean_ci": [float(mlo), float(mhi)]}
    print(f"{name:<17}{v.max()/v.min():8.2f}{f'[{lo:.2f}, {hi:.2f}]':>18}"
          f"{v.mean():11.2f}{f'[{mlo:.2f}, {mhi:.2f}]':>18}")

# do the spread CIs of the compared models overlap?
print("\n=== overlap of spread CIs with the headline ===")
h = res.get("PoseNetTx")
for k, r in res.items():
    if k == "PoseNetTx": continue
    ov = not (r["spread_ci"][1] < h["spread_ci"][0] or r["spread_ci"][0] > h["spread_ci"][1])
    print(f"  {k:<17} {'OVERLAPS (not distinguishable)' if ov else 'DISJOINT (distinguishable)'}")
json.dump(res, open(f"{OUT}/bootstrap_ci.json", "w"), indent=2)
print("\nwrote bootstrap_ci.json")
