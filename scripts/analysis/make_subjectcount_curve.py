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
"""Assemble the subject-count curve from the N-subject runs + the full headline."""
import json, os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = str(_ROOT / "experiments")
OUT = str(_THESIS)
SEC = os.path.join(OUT, "06_subject_count")
for d in (SEC, os.path.join(OUT,"plots"), os.path.join(OUT,"reports"), os.path.join(OUT,"data_tables")):
    os.makedirs(d, exist_ok=True)

points = []   # (N, mean, std, best, worst, spread)
# N<17 runs: tag v4scNN
for p in glob.glob(f"{E}/*posenet_tx*v4sc*/loso_summary.json"):
    m = re.search(r"v4sc(\d+)", p)
    if not m: continue
    N = int(m.group(1))
    d = json.load(open(p))
    if d.get("n_folds",0) < 21: continue
    mae = [f["test_mae"] for f in d["per_fold"]]
    points.append((N, d["mean_test_mae"], d["std_test_mae"], min(mae), max(mae), max(mae)/min(mae)))
# N=17 full headline
hp = f"{E}/run28_posenet_tx_lr3e-4_bs64_ep60_loso_v4headline/loso_summary.json"
if os.path.exists(hp):
    d = json.load(open(hp)); mae=[f["test_mae"] for f in d["per_fold"]]
    points.append((17, d["mean_test_mae"], d["std_test_mae"], min(mae), max(mae), max(mae)/min(mae)))

points.sort()
if not points:
    print("no subject-count runs found yet"); raise SystemExit

Ns   = [p[0] for p in points]
mean = [p[1] for p in points]; std=[p[2] for p in points]
best = [p[3] for p in points]; worst=[p[4] for p in points]; spread=[p[5] for p in points]

# --- Figure 1: spread vs N (the key plot) ---
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(Ns, spread, "o-", color="#b2182b", lw=2, ms=8)
for x,y in zip(Ns,spread): ax.annotate(f"{y:.2f}x",(x,y),textcoords="offset points",xytext=(0,8),ha="center",fontsize=9)
ax.set_xlabel("Number of training subjects (N)"); ax.set_ylabel("Cross-subject spread (worst / best)")
ax.set_title("Does cross-subject spread shrink with more training subjects?")
ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/plots/subjectcount_spread_vs_N.png", dpi=150); plt.close()

# --- Figure 2: mean + worst + best vs N ---
fig, ax = plt.subplots(figsize=(8,5))
ax.errorbar(Ns, mean, yerr=std, fmt="o-", color="#2166ac", lw=2, capsize=4, label="mean MAE")
ax.plot(Ns, worst, "s--", color="#b2182b", lw=1.5, label="worst subject")
ax.plot(Ns, best,  "^--", color="#2e7d32", lw=1.5, label="best subject")
ax.set_xlabel("Number of training subjects (N)"); ax.set_ylabel("Test MAE (degrees)")
ax.set_title("Error vs training-subject count"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/plots/subjectcount_mae_vs_N.png", dpi=150); plt.close()

# --- table + report ---
import csv
with open(f"{OUT}/data_tables/subjectcount.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["N_train","mean_MAE","std","best","worst","spread"])
    for p in points: w.writerow([p[0],f"{p[1]:.3f}",f"{p[2]:.3f}",f"{p[3]:.2f}",f"{p[4]:.2f}",f"{p[5]:.3f}"])
json.dump({"points":[{"N":p[0],"mean":p[1],"std":p[2],"best":p[3],"worst":p[4],"spread":p[5]} for p in points]},
          open(f"{SEC}/subjectcount_summary.json","w"), indent=2)

lines=["# Subject-count curve (PoseNetTx, loso_v4, full LOSO)","",
       "| N train | mean MAE | best | worst | spread |","|---|---|---|---|---|"]
for p in points: lines.append(f"| {p[0]} | {p[1]:.3f} | {p[3]:.2f} | {p[4]:.2f} | {p[5]:.2f}x |")
trend = spread[0]-spread[-1]
lines += ["", f"Spread from N={Ns[0]} to N={Ns[-1]}: {spread[0]:.2f}x -> {spread[-1]:.2f}x "
          f"({'DECREASES' if trend>0.05 else 'FLAT' if abs(trend)<=0.05 else 'INCREASES'} with more subjects).",
          "", "![spread](../plots/subjectcount_spread_vs_N.png)","",
          "![mae](../plots/subjectcount_mae_vs_N.png)"]
open(f"{SEC}/subjectcount_report.md","w").write("\n".join(lines))
print("subject-count curve done. points:", points)
