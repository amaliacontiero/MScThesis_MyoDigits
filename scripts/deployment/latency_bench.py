"""
Inference latency benchmark for MyoDigits models.

Measures GPU and CPU latency in three deployment scenarios:
  1. Block mode     — process a full T-window sequence at once: (1, T, C, 400) → (1, T, 20)
                      relevant when BiGRU requires the full context (PoseNetSeq)
  2. Single-window  — process one window independently: (1, 1, C, 400) → (1, 1, 20)
                      relevant for TCN-based models that are window-independent
  3. Batch          — throughput at batch sizes [1, 8, 32, 64]

Real-time constraint: 50 ms per window (2000 Hz, 400-sample stride).

Usage:
    python scripts/latency_bench.py
    python scripts/latency_bench.py --ckpt experiments/run06_.../fold01_.../checkpoints/best.pt --model posenet_seq --in_ch 8
    python scripts/latency_bench.py --model tcn_v1 --in_ch 16  # random weights, no checkpoint
"""

import sys
import argparse
import time
import importlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RT_BUDGET_MS = 50.0   # one 50-ms EMG window

MODEL_REGISTRY = {
    "posenet_seq": ("myodigits.models.posenet_seq", "PoseNetSeq"),
    "tcn_v1":      ("myodigits.models.tcn_v1",      "PoseNetTCN"),
    "posenet_tx":  ("myodigits.models.posenet_tx",  "PoseNetTx"),
    "cnn_v1":      ("myodigits.models.cnn_v1",      "CNN"),
    "lstm_v1":     ("myodigits.models.lstm_v1",      "LSTM"),
}

WARMUP  = 50    # forward passes to warm up CUDA / JIT
REPEATS = 500   # timed passes


def _load_model(model_name, in_ch, ckpt_path, device):
    module_path, class_name = MODEL_REGISTRY[model_name]
    ModelClass = getattr(importlib.import_module(module_path), class_name)

    import inspect
    kwargs = {}
    sig = inspect.signature(ModelClass.__init__)
    if "in_ch" in sig.parameters:
        kwargs["in_ch"] = in_ch

    model = ModelClass(**kwargs).to(device)
    model.eval()

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded: {ckpt_path}  (epoch={ckpt.get('epoch','?')}, "
              f"val_mae={ckpt.get('best_val_mae', float('nan')):.4f})")
    else:
        print("No checkpoint — using random weights (architecture timing only)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {class_name}  |  params: {n_params:,}\n")
    return model


def _bench(model, x, device, label, budget_ms=None):
    """
    Run WARMUP + REPEATS forward passes, return stats dict.
    x : tensor already on device
    """
    with torch.no_grad():
        # warm-up
        for _ in range(WARMUP):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(REPEATS):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000   # ms total

    mean_ms = elapsed / REPEATS
    std_ms  = 0.0   # single-threaded timing — we report mean only

    marker = ""
    if budget_ms is not None:
        marker = "✓ REAL-TIME" if mean_ms <= budget_ms else f"✗ OVER by {mean_ms - budget_ms:.1f} ms"

    print(f"  {label:<40s}  {mean_ms:7.3f} ms / call   {marker}")
    return {"label": label, "mean_ms": mean_ms, "budget_ms": budget_ms}


def _bench_fp16(model, x, device, label):
    """AMP autocast forward pass (GPU only)."""
    if device.type != "cuda":
        return None
    with torch.no_grad():
        for _ in range(WARMUP):
            with torch.amp.autocast("cuda"):
                _ = model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(REPEATS):
            with torch.amp.autocast("cuda"):
                _ = model(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

    mean_ms = elapsed / REPEATS
    print(f"  {label:<40s}  {mean_ms:7.3f} ms / call   (fp16/AMP)")
    return {"label": label, "mean_ms": mean_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default=None, help="Path to best.pt checkpoint")
    parser.add_argument("--model",   default="posenet_seq")
    parser.add_argument("--in_ch",   default=8, type=int)
    parser.add_argument("--seq_len", default=64, type=int, help="T (windows per sequence)")
    parser.add_argument("--cpu_only",action="store_true")
    args = parser.parse_args()

    device_gpu = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu_only) else "cpu")
    device_cpu = torch.device("cpu")

    C, W = args.in_ch, 400
    T    = args.seq_len

    print("=" * 65)
    print(f"MyoDigits Latency Benchmark — {args.model}  in_ch={C}  seq_len={T}")
    if device_gpu.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Real-time budget: {RT_BUDGET_MS} ms / window")
    print("=" * 65)

    # ── GPU benchmarks ────────────────────────────────────────────────────────
    if device_gpu.type == "cuda":
        model_gpu = _load_model(args.model, C, args.ckpt, device_gpu)

        print("\n--- GPU (fp32) ---")

        # 1. Block mode: full T-window sequence, batch=1
        x_block = torch.randn(1, T, C, W, device=device_gpu)
        r = _bench(model_gpu, x_block, device_gpu,
                   f"block(1,{T},{C},400) → per-window",
                   budget_ms=RT_BUDGET_MS * T)
        # effective per-window latency inside the block
        eff_ms = r["mean_ms"] / T
        print(f"    └─ effective per-window: {eff_ms:.3f} ms  "
              f"({'✓' if eff_ms <= RT_BUDGET_MS else '✗'} vs {RT_BUDGET_MS} ms budget)")

        # 2. Single-window: (1, 1, C, W) — only meaningful for window-independent models
        x_single = torch.randn(1, 1, C, W, device=device_gpu)
        _bench(model_gpu, x_single, device_gpu,
               f"single(1,1,{C},400) 1-window",
               budget_ms=RT_BUDGET_MS)

        # 3. Throughput at various batch sizes (block mode)
        print("\n--- GPU throughput (block mode, fp32) ---")
        for bs in [1, 8, 32, 64]:
            x_bs = torch.randn(bs, T, C, W, device=device_gpu)
            r = _bench(model_gpu, x_bs, device_gpu,
                       f"batch={bs:<3d}  ({bs},{T},{C},400)")
            throughput = bs * T / (r["mean_ms"] / 1000)
            print(f"    └─ throughput: {throughput:,.0f} windows/s")

        # 4. AMP (fp16)
        print("\n--- GPU (fp16 / AMP) ---")
        x_block_gpu = torch.randn(1, T, C, W, device=device_gpu)
        _bench_fp16(model_gpu, x_block_gpu, device_gpu,
                    f"block(1,{T},{C},400) fp16")

        del model_gpu
        torch.cuda.empty_cache()

    # ── CPU benchmarks ────────────────────────────────────────────────────────
    print("\n--- CPU (fp32) ---")
    model_cpu = _load_model(args.model, C, args.ckpt, device_cpu)

    x_block_cpu = torch.randn(1, T, C, W)
    r = _bench(model_cpu, x_block_cpu, device_cpu,
               f"block(1,{T},{C},400) → per-window")
    eff_ms_cpu = r["mean_ms"] / T
    print(f"    └─ effective per-window: {eff_ms_cpu:.3f} ms  "
          f"({'✓' if eff_ms_cpu <= RT_BUDGET_MS else '✗'} vs {RT_BUDGET_MS} ms budget)")

    x_single_cpu = torch.randn(1, 1, C, W)
    _bench(model_cpu, x_single_cpu, device_cpu,
           f"single(1,1,{C},400) 1-window",
           budget_ms=RT_BUDGET_MS)

    # ── Deployment notes ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Deployment notes")
    print("=" * 65)
    if args.model == "posenet_seq":
        print(f"""
  PoseNetSeq uses a BIDIRECTIONAL GRU — it needs the entire T={T}-window
  block before producing any output (non-causal).

  Deployment options:
    A) Block-latency mode: buffer 64 windows (3.2 s), run one forward pass,
       output all 64 predictions at once. Total latency = 3.2 s + GPU time.
       Good for offline / post-hoc analysis.

    B) Causal variant: replace BiGRU with unidirectional GRU.
       Each new window triggers one forward pass with running hidden state.
       Latency = single-window GPU time ≈ block_time / {T}.
       Loses the backward context (slight accuracy drop expected).

    C) TCN-only (PoseNetTCN / run08): fully causal by design.
       Single-window latency with zero sequence-level state.
""")
    elif args.model in ("tcn_v1", "cnn_v1"):
        print(f"""
  {args.model} is window-independent — fully causal / streaming-ready.
  Single-window latency is the direct deployment latency.
""")
    elif args.model == "posenet_tx":
        print(f"""
  PoseNetTx Transformer is non-causal (full self-attention over T={T} windows).
  Same block-mode constraint as PoseNetSeq.
  For streaming: use causal masking (causal_mask=True in TransformerEncoder).
""")


if __name__ == "__main__":
    main()
