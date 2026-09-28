#!/usr/bin/env python3
"""INT8 dynamic quantisation under the controlled CPU protocol.
Reports serialized size and CPU latency at pinned thread counts, and reports
failure explicitly rather than omitting a model."""
# --- configurable roots -------------------------------------------------
# The analysis and figure scripts read run directories that are not
# distributed with this repository. Point these at your own copies.
import os as _os
from pathlib import Path as _Path
_ROOT   = _Path(_os.environ.get("MYODIGITS_ROOT", _Path(__file__).resolve().parents[2]))
_THESIS = _Path(_os.environ.get("MYODIGITS_THESIS_DIR", _ROOT))
_EGOEMG = _Path(_os.environ.get("MYODIGITS_EGOEMG_DIR", _ROOT.parent / "MD_EgoEMG"))
# ------------------------------------------------------------------------
import os, sys, json, time, tempfile, traceback
import numpy as np, torch, torch.nn as nn

os.chdir(str(_ROOT)); sys.path.insert(0, "src")
NAME = sys.argv[1]
SEQ, IN_CH, WIN = 64, 16, 400
CPU_N, CPU_WARM = 500, 100

from myodigits.models.posenet_tx import PoseNetTx
from myodigits.models.posenet_seq import PoseNetSeq
from myodigits.models.tcn_v1 import PoseNetTCN
BUILD = {"MyoDigits-TX": lambda: PoseNetTx(in_ch=IN_CH, out_dim=20, base=48, tx_heads=4,
                                           tx_layers=2, tx_dim_ff=256),
         "MyoDigits-GRU": lambda: PoseNetSeq(in_ch=IN_CH, out_dim=20, base=48),
         "MyoDigits-PW": lambda: PoseNetTCN(in_ch=IN_CH, out_dim=20, base=48)}

def ser_mb(mod):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(mod.state_dict(), f.name); s = os.path.getsize(f.name) / 1e6
    os.unlink(f.name); return s

def bench(fn, n, warm):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)), n=n, warmup=warm)

x = torch.randn(1, SEQ, IN_CH, WIN)
res = {"model": NAME}
m = BUILD[NAME]().eval()
res["fp32_serialized_mb"] = ser_mb(m)
res["raw_param_storage_mb"] = sum(p.numel() for p in m.parameters()) * 4 / 1e6

for nt in (1, 8):
    torch.set_num_threads(nt)
    res[f"fp32_cpu_{nt}t"] = bench(lambda: m(x), CPU_N, CPU_WARM)

try:
    qm = torch.ao.quantization.quantize_dynamic(m, {nn.Linear, nn.GRU}, dtype=torch.qint8).eval()
    res["int8_serialized_mb"] = ser_mb(qm)
    res["int8_quantised_modules"] = sum(
        1 for _, mm in qm.named_modules()
        if "quantized" in type(mm).__module__.lower() or "Dynamic" in type(mm).__name__)
    try:
        with torch.no_grad(): qm(x)
        for nt in (1, 8):
            torch.set_num_threads(nt)
            res[f"int8_cpu_{nt}t"] = bench(lambda: qm(x), CPU_N, CPU_WARM)
        res["int8_latency_status"] = "measured"
    except Exception as e:
        res["int8_latency_status"] = "FAILED"
        res["int8_latency_error"] = f"{type(e).__name__}: {e}"
        res["int8_latency_traceback_tail"] = traceback.format_exc().strip().split("\n")[-3:]
except Exception as e:
    res["int8_status"] = "QUANTISATION FAILED"
    res["int8_error"] = f"{type(e).__name__}: {e}"

print(json.dumps(res, indent=1))
