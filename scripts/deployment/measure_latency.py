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
"""Measure inference latency ONLY on an idle GPU. Refuses to run under contention."""
import sys, os, time, json, subprocess
sys.path.insert(0, str(_ROOT / "src"))

OUT = str(_THESIS / "figures/latency.json")

def gpu_busy():
    """Contention gate. The hard requirement is that no OTHER CUDA compute process
    is running. This host shows an irreducible 11-16% utilization from Windows
    desktop/display rendering even with zero CUDA processes, so a utilization
    threshold alone is not a valid idleness test; it is recorded, not gated on
    (a generous ceiling is kept to catch a genuinely saturated device)."""
    try:
        util = int(subprocess.check_output(
            ["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"]
        ).decode().strip().split("\n")[0])
        procs = subprocess.check_output(
            ["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"]
        ).decode().strip()
        n_other = len([p for p in procs.split("\n") if p.strip()])
    except Exception as e:
        print("cannot query GPU:", e); return True, -1, -1
    return (n_other > 0 or util > 40), util, n_other

busy, util, nproc = gpu_busy()
if busy:
    print(f"ABORT: GPU not idle ({nproc} competing CUDA process(es), utilization {util}%).")
    print("Latency under contention is not a valid measurement. Re-run when the queue is empty.")
    sys.exit(1)
print(f"GPU clear: {nproc} competing CUDA processes, {util}% baseline (display rendering).")

import torch
from myodigits.models.posenet_tx import PoseNetTx

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = PoseNetTx(in_ch=16, out_dim=20, base=48, tx_heads=4, tx_layers=2, tx_dim_ff=256).to(dev).eval()
nparam = sum(p.numel() for p in model.parameters())
res = {"params": nparam, "device": dev, "gpu_util_before": util}

def bench(T, reps=200, warmup=50):
    x = torch.randn(1, T, 16, 400, device=dev)
    with torch.no_grad():
        for _ in range(warmup): model(x)
        if dev == "cuda": torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            if dev == "cuda": torch.cuda.synchronize()
            ts.append((time.perf_counter()-t0)*1000)
    ts.sort()
    return {"median_ms": ts[len(ts)//2], "p95_ms": ts[int(0.95*len(ts))],
            "min_ms": ts[0], "mean_ms": sum(ts)/len(ts)}

for label, T in [("single_window_200ms", 1), ("sequence_64win_3.2s", 64)]:
    r = bench(T)
    res[label] = r
    print(f"{label:22s} median {r['median_ms']:.3f} ms  p95 {r['p95_ms']:.3f} ms  min {r['min_ms']:.3f} ms")

busy_after, util_after, _ = gpu_busy()
res["gpu_util_after"] = util_after
res["valid"] = True
# sanity: sequence should cost meaningfully more than a single window
ratio = res["sequence_64win_3.2s"]["median_ms"] / res["single_window_200ms"]["median_ms"]
res["seq_over_single_ratio"] = ratio
print(f"sequence/single ratio = {ratio:.2f} (expect > 1; near 1.0 suggests overhead-dominated)")
res["realtime_factor"] = 200.0 / res["single_window_200ms"]["median_ms"]
print(f"real-time factor (200ms window / inference) = {res['realtime_factor']:.1f}x")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print("wrote", OUT)
