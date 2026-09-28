#!/usr/bin/env python3
"""Deployment profiling, controlled protocol (v2).

Supersedes measure_cost.py / measure_deployment.py. Fixes, in order:
  1. CPU thread count explicitly set and reported; measured at 1 and at 8 threads.
  2. Environment fully recorded (CPU model, GPU model, torch/CUDA, device state).
  3. Larger samples: GPU n=1000/warmup=200, CPU n=500/warmup=100, so p95 is meaningful.
  4. GPU and CPU sample counts reported separately, never conflated.
  5. MACs counted with the MHA fastpath DISABLED so the Transformer is included,
     cross-checked against train-mode counting; per-module breakdown retained.
  6. Raw parameter storage, serialized checkpoint size and GPU memory are three
     separate quantities and are never merged.
  7. Activations isolated by subtracting resident weights+input from the peak.
  8. The 22.0 vs 32.56 MB discrepancy is investigated directly.
  9. Length-one forward is labelled as such. No streaming implementation exists,
     so no streaming number is produced.
"""
# --- configurable roots -------------------------------------------------
# The analysis and figure scripts read run directories that are not
# distributed with this repository. Point these at your own copies.
import os as _os
from pathlib import Path as _Path
_ROOT   = _Path(_os.environ.get("MYODIGITS_ROOT", _Path(__file__).resolve().parents[2]))
_THESIS = _Path(_os.environ.get("MYODIGITS_THESIS_DIR", _ROOT))
_EGOEMG = _Path(_os.environ.get("MYODIGITS_EGOEMG_DIR", _ROOT.parent / "MD_EgoEMG"))
# ------------------------------------------------------------------------
import os, sys, json, platform, subprocess, tempfile, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

ROOT = str(_ROOT)
os.chdir(ROOT); sys.path.insert(0, "src")
OUT = str(_THESIS / "figures2/deployment_v2.json")

SEQ, IN_CH, WIN, HOP_MS = 64, 16, 400, 50.0
GPU_N, GPU_WARM = 1000, 200
CPU_N, CPU_WARM = 500, 100

from myodigits.models.posenet_tx import PoseNetTx
from myodigits.models.posenet_seq import PoseNetSeq
from myodigits.models.tcn_v1 import PoseNetTCN

def build(name):
    if name == "MyoDigits-TX":
        return PoseNetTx(in_ch=IN_CH, out_dim=20, base=48, tx_heads=4, tx_layers=2, tx_dim_ff=256)
    if name == "MyoDigits-GRU":
        return PoseNetSeq(in_ch=IN_CH, out_dim=20, base=48)
    if name == "MyoDigits-PW":
        return PoseNetTCN(in_ch=IN_CH, out_dim=20, base=48)
NAMES = ["MyoDigits-TX", "MyoDigits-GRU", "MyoDigits-PW"]

# ---------------------------------------------------------------- environment
def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"): return line.split(":", 1)[1].strip()
    except Exception: pass
    return platform.processor() or "unknown"

def gpu_state():
    try:
        q = "utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.current.sm"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out
    except Exception as e:
        return f"unavailable ({e})"

ENV = {
    "cpu_model": cpu_model(),
    "cpu_logical_cores": os.cpu_count(),
    "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "platform": platform.platform(),
    "gpu_state_before": gpu_state(),
}
print("=" * 78); print("ENVIRONMENT"); print("=" * 78)
for k, v in ENV.items(): print(f"  {k:22s} {v}")

# ------------------------------------------------------------------- timing
def bench(fn, n, warmup, cuda):
    for _ in range(warmup): fn()
    if cuda: torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if cuda: torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return dict(median=float(np.median(a)), mean=float(a.mean()),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                min=float(a.min()), max=float(a.max()), n=n, warmup=warmup)

# --------------------------------------------------------------------- MACs
def count_macs(m, x, label):
    """Count with MHA fastpath disabled so Transformer ops are visible."""
    with torch.no_grad():
        fc = FlopCounterMode(display=False, depth=2)
        with fc: m(x)
        total = fc.get_total_flops() / 2.0
        per_mod = {k: sum(v.values()) / 2.0 for k, v in fc.get_flop_counts().items()}
    return total, per_mod

have_fastpath_switch = hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled")
print(f"\n  torch.backends.mha.set_fastpath_enabled available: {have_fastpath_switch}")

results = {}
x_seq = torch.randn(1, SEQ, IN_CH, WIN)
x_one = torch.randn(1, 1, IN_CH, WIN)

for name in NAMES:
    print("\n" + "=" * 78); print(name); print("=" * 78)
    m = build(name).eval()
    r = {}

    # --- storage, three distinct quantities -------------------------------
    n_par = sum(p.numel() for p in m.parameters())
    n_buf = sum(b.numel() for b in m.buffers())
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(m.state_dict(), f.name); ser = os.path.getsize(f.name) / 1e6
    os.unlink(f.name)
    r["params"] = n_par
    r["buffers"] = n_buf
    r["raw_param_storage_mb"] = n_par * 4 / 1e6
    r["raw_param_plus_buffer_mb"] = (n_par + n_buf) * 4 / 1e6
    r["serialized_state_dict_mb"] = ser
    print(f"  params {n_par:,}  buffers {n_buf:,}")
    print(f"  raw parameter storage      : {r['raw_param_storage_mb']:.6f} MB")
    print(f"  raw params+buffers         : {r['raw_param_plus_buffer_mb']:.6f} MB")
    print(f"  serialized state_dict file : {ser:.6f} MB")

    # --- MACs, fastpath disabled ------------------------------------------
    if have_fastpath_switch:
        torch.backends.mha.set_fastpath_enabled(False)
    macs_fp_off, per_mod = count_macs(m, x_seq, "fastpath-off")
    if have_fastpath_switch:
        torch.backends.mha.set_fastpath_enabled(True)
    macs_fp_on, _ = count_macs(m, x_seq, "fastpath-on")
    macs_train, _ = count_macs(build(name).train(), x_seq, "train")
    r["macs_fastpath_disabled"] = macs_fp_off
    r["macs_fastpath_enabled"] = macs_fp_on
    r["macs_train_mode_crosscheck"] = macs_train
    r["macs_reported"] = max(macs_fp_off, macs_train)
    print(f"  MACs fastpath OFF : {macs_fp_off:,.0f}")
    print(f"  MACs fastpath ON  : {macs_fp_on:,.0f}   (the old, blind measurement)")
    print(f"  MACs train-mode   : {macs_train:,.0f}   (independent cross-check)")
    conv = per_mod.get("Global", {})
    # decoder share: total minus the stem+blocks+head convolution mass
    r["macs_per_module_top"] = dict(sorted(
        ((k, v) for k, v in per_mod.items() if k != "Global" and "." not in k.split(".", 1)[-1][:0] + k[len(name):]),
        key=lambda kv: -kv[1])[:1])

    # --- latency, CPU at controlled thread counts -------------------------
    r["cpu"] = {}
    for nthreads in (1, 8):
        torch.set_num_threads(nthreads)
        mc = m.cpu()
        r["cpu"][f"{nthreads}_threads"] = {
            "threads_set": nthreads,
            "threads_effective": torch.get_num_threads(),
            "seq_ms": bench(lambda: mc(x_seq), CPU_N, CPU_WARM, False),
            "single_window_ms": bench(lambda: mc(x_one), CPU_N, CPU_WARM, False),
        }
        s = r["cpu"][f"{nthreads}_threads"]["seq_ms"]
        w = r["cpu"][f"{nthreads}_threads"]["single_window_ms"]
        print(f"  CPU {nthreads} thread(s): seq median {s['median']:.2f} p95 {s['p95']:.2f} ms "
              f"(n={s['n']}) | 1-window median {w['median']:.3f} p95 {w['p95']:.3f} ms")
    torch.set_num_threads(8)

    # --- GPU latency and memory, three separated quantities ---------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        mg = m.cuda().eval(); xg = x_seq.cuda(); xo = x_one.cuda()
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated() / 1e6          # weights + input
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(): mg(xg)
        torch.cuda.synchronize()
        peak_1 = torch.cuda.max_memory_allocated() / 1e6
        r["gpu_resident_weights_plus_input_mb"] = resident
        r["gpu_peak_total_after_1_forward_mb"] = peak_1
        r["gpu_activations_isolated_mb"] = peak_1 - resident
        print(f"  GPU resident (weights+input) : {resident:.3f} MB")
        print(f"  GPU peak total, 1 forward    : {peak_1:.3f} MB")
        print(f"  GPU activations (isolated)   : {peak_1-resident:.3f} MB")

        g = bench(lambda: mg(xg), GPU_N, GPU_WARM, True)
        torch.cuda.synchronize()
        peak_many = torch.cuda.max_memory_allocated() / 1e6
        r["gpu_peak_total_after_bench_mb"] = peak_many
        r["gpu_seq_ms"] = g
        print(f"  GPU peak total, after {GPU_N} runs: {peak_many:.3f} MB   "
              f"<-- discrepancy probe (1-run vs many-run)")
        print(f"  GPU seq: median {g['median']:.3f}  p95 {g['p95']:.3f}  p99 {g['p99']:.3f} ms (n={g['n']})")
        r["gpu_single_window_ms"] = bench(lambda: mg(xo), GPU_N, GPU_WARM, True)
        m = m.cpu(); del mg, xg, xo; torch.cuda.empty_cache()

    # --- budget: arithmetic derivation, explicitly labelled ---------------
    cpu8 = r["cpu"]["8_threads"]["seq_ms"]["median"]
    r["hop_budget_ms"] = HOP_MS
    r["budget_fraction_cpu8_median_DERIVED"] = cpu8 / HOP_MS
    r["note_budget"] = ("Arithmetic derivation from the measured CPU median, not a "
                        "measurement. Assumes full-context recompute every hop.")
    r["note_streaming"] = ("No stateful streaming implementation exists. The single-window "
                           "figure is a length-one forward pass, NOT a streaming benchmark.")
    results[name] = r

ENV["gpu_state_after"] = gpu_state()
payload = {"meta": dict(seq_len=SEQ, in_ch=IN_CH, win_samples=WIN, hop_ms=HOP_MS,
                        gpu_n=GPU_N, gpu_warmup=GPU_WARM, cpu_n=CPU_N, cpu_warmup=CPU_WARM,
                        environment=ENV,
                        protocol="controlled: threads pinned, fastpath disabled for MACs, "
                                 "activations isolated, storage reported three ways"),
           "models": results}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(payload, open(OUT, "w"), indent=1)
print(f"\n\nwrote {OUT}")
