#!/usr/bin/env python3
"""One model, one fresh process. Removes the cross-model carry-over that
contaminated the in-process run, and probes the peak-memory growth directly."""
# --- configurable roots -------------------------------------------------
# The analysis and figure scripts read run directories that are not
# distributed with this repository. Point these at your own copies.
import os as _os
from pathlib import Path as _Path
_ROOT   = _Path(_os.environ.get("MYODIGITS_ROOT", _Path(__file__).resolve().parents[2]))
_THESIS = _Path(_os.environ.get("MYODIGITS_THESIS_DIR", _ROOT))
_EGOEMG = _Path(_os.environ.get("MYODIGITS_EGOEMG_DIR", _ROOT.parent / "MD_EgoEMG"))
# ------------------------------------------------------------------------
import os, sys, json, time
import numpy as np, torch

ROOT = str(_ROOT)
os.chdir(ROOT); sys.path.insert(0, "src")
NAME = sys.argv[1]
SEQ, IN_CH, WIN = 64, 16, 400

from myodigits.models.posenet_tx import PoseNetTx
from myodigits.models.posenet_seq import PoseNetSeq
from myodigits.models.tcn_v1 import PoseNetTCN
m = {"MyoDigits-TX": lambda: PoseNetTx(in_ch=IN_CH, out_dim=20, base=48, tx_heads=4,
                                       tx_layers=2, tx_dim_ff=256),
     "MyoDigits-GRU": lambda: PoseNetSeq(in_ch=IN_CH, out_dim=20, base=48),
     "MyoDigits-PW": lambda: PoseNetTCN(in_ch=IN_CH, out_dim=20, base=48)}[NAME]().eval()

x = torch.randn(1, SEQ, IN_CH, WIN)
res = {"model": NAME}

# ---- CPU latency, threads pinned, large n -------------------------------
def bench(fn, n, warm):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), n=n, warmup=warm)

for nt in (1, 8):
    torch.set_num_threads(nt)
    res[f"cpu_{nt}t_seq"] = bench(lambda: m(x), 500, 100)
    res[f"cpu_{nt}t_win"] = bench(lambda: m(torch.randn(1, 1, IN_CH, WIN)), 300, 60)

# ---- GPU memory, clean process, growth probe ----------------------------
if torch.cuda.is_available():
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    mg = m.cuda().eval()
    torch.cuda.synchronize()
    w_only = torch.cuda.memory_allocated() / 1e6
    xg = x.cuda(); torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(): mg(xg)
    torch.cuda.synchronize()
    p1 = torch.cuda.max_memory_allocated() / 1e6
    res["gpu_weights_only_mb"] = w_only
    res["gpu_resident_weights_plus_input_mb"] = resident
    res["gpu_peak_1_forward_mb"] = p1
    res["gpu_activations_1_forward_mb"] = p1 - resident
    # growth probe: peak as a function of iteration count
    growth = {}
    for k in (1, 10, 50, 240, 1000):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(k): mg(xg)
        torch.cuda.synchronize()
        growth[k] = torch.cuda.max_memory_allocated() / 1e6
    res["gpu_peak_vs_iterations_mb"] = growth
    torch.cuda.reset_peak_memory_stats()
    g = []
    with torch.no_grad():
        for _ in range(200): mg(xg)
        torch.cuda.synchronize()
        for _ in range(1000):
            t0 = time.perf_counter(); mg(xg); torch.cuda.synchronize()
            g.append((time.perf_counter() - t0) * 1e3)
    a = np.array(g)
    res["gpu_seq"] = dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                          p99=float(np.percentile(a, 99)), n=1000, warmup=200)

print(json.dumps(res, indent=1))
