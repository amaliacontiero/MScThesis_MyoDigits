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
"""How much computation does this thesis actually rest on?

Wall-clock training time is reconstructed from the modification times of each
fold's metrics.json, which is written when the fold finishes. Consecutive
differences inside a run give per-fold durations; gaps longer than SANE_MAX are
pauses (the machine was off, or the queue was idle) and are excluded rather
than counted as training.

Some experiment directories carry a single restore timestamp rather than their
original times. Those runs are reported as fold counts only, never as hours,
and the distinction is stated in the output.

Writes figures2/effort.json.
"""
import json, glob, os, re
from collections import defaultdict
import numpy as np

EXP = str(_ROOT / "experiments")
OUT = str(_THESIS / "figures2/effort.json")
SANE_MAX = 12 * 3600      # a single fold never legitimately takes longer
RESTORE_WINDOW = 120      # folds whose times cluster this tightly are a restore

runs = {}
for d in sorted(os.listdir(EXP)):
    p = os.path.join(EXP, d)
    if not d.startswith("run") or not os.path.isdir(p):
        continue
    ms = sorted(glob.glob(f"{p}/fold*_testUser*/metrics.json"), key=os.path.getmtime)
    if not ms:
        continue
    ts = [os.path.getmtime(m) for m in ms]
    spread = ts[-1] - ts[0]
    trustworthy = not (len(ts) > 2 and spread < RESTORE_WINDOW)
    gaps = [b - a for a, b in zip(ts, ts[1:]) if 0 < (b - a) < SANE_MAX]
    runs[d] = dict(folds=len(ms), trustworthy=trustworthy,
                   hours=sum(gaps) / 3600 if trustworthy else None,
                   per_fold_h=(np.median(gaps) / 3600) if (trustworthy and gaps) else None,
                   first=ts[0], last=ts[-1])

meas = {k: v for k, v in runs.items() if v["trustworthy"] and v["hours"]}
unmeas = {k: v for k, v in runs.items() if k not in meas}

tot_folds = sum(v["folds"] for v in runs.values())
tot_hours_measured = sum(v["hours"] for v in meas.values())
# folds in runs whose clock is unusable, costed at the measured median rate
rates = [v["per_fold_h"] for v in meas.values() if v["per_fold_h"]]
med_rate = float(np.median(rates)) if rates else 3.0
unmeas_folds = sum(v["folds"] for v in unmeas.values())
est_unmeas_hours = unmeas_folds * med_rate

print(f"{'run':52s} {'folds':>5} {'hours':>8} {'h/fold':>7}")
for k, v in sorted(runs.items()):
    h = f"{v['hours']:8.1f}" if v["hours"] else "       -"
    pf = f"{v['per_fold_h']:7.2f}" if v["per_fold_h"] else "      -"
    print(f"{k[:52]:52s} {v['folds']:5d} {h} {pf}")

print(f"\nruns                       {len(runs)}")
print(f"folds trained              {tot_folds}")
print(f"  with usable timestamps   {sum(v['folds'] for v in meas.values())}")
print(f"  timestamps unusable      {unmeas_folds}")
print(f"median time per fold       {med_rate:.2f} h")
print(f"measured GPU time          {tot_hours_measured:.0f} h "
      f"({tot_hours_measured/24:.1f} days)")
print(f"estimated for the rest     {est_unmeas_hours:.0f} h")
print(f"TOTAL, measured + estimate {tot_hours_measured + est_unmeas_hours:.0f} h "
      f"({(tot_hours_measured + est_unmeas_hours)/24:.1f} days)")

# checkpoint footprint
ck = glob.glob(f"{EXP}/run*/fold*/checkpoints/*.pt")
gb = sum(os.path.getsize(c) for c in ck) / 1e9
print(f"\ncheckpoints                {len(ck)} files, {gb:.1f} GB")

json.dump(dict(runs={k: {kk: vv for kk, vv in v.items() if kk not in ('first', 'last')}
                     for k, v in runs.items()},
               total_runs=len(runs), total_folds=tot_folds,
               measured_hours=tot_hours_measured,
               estimated_hours=est_unmeas_hours,
               total_hours=tot_hours_measured + est_unmeas_hours,
               median_fold_hours=med_rate,
               checkpoints=len(ck), checkpoint_gb=gb),
          open(OUT, "w"), indent=1)
print("\nwrote", OUT)
