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
"""Computational cost of each definitive architecture.

Reports parameters, multiply-accumulate operations per forward pass, model size,
peak inference memory, and forward latency on GPU and CPU. Latency is measured
on an otherwise idle device; a measurement taken while the GPU is training is
inflated several-fold and is not reported here.

Writes figures2/model_cost.json for the thesis builder.
"""
import json, os, sys, time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(_ROOT / "src"))
OUT = str(_THESIS / "figures2/model_cost.json")

SEQ, IN_CH, WIN = 64, 16, 400          # 64 windows of 400 samples, 16 channels

def build(name):
    if name == "MyoDigits-TX":
        from myodigits.models.posenet_tx import PoseNetTx
        return PoseNetTx(in_ch=IN_CH, out_dim=20, base=48,
                         tx_heads=4, tx_layers=2, tx_dim_ff=256)
    if name == "MyoDigits-GRU":
        from myodigits.models.posenet_seq import PoseNetSeq
        return PoseNetSeq(in_ch=IN_CH, out_dim=20, base=48)
    if name == "MyoDigits-PW":
        from myodigits.models.tcn_v1 import PoseNetTCN
        return PoseNetTCN(in_ch=IN_CH, out_dim=20, base=48)
    raise ValueError(name)


def bench(m, x, dev, n=200, warmup=40):
    m = m.to(dev).eval()
    x = x.to(dev)
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            m(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used_mb = (total - free) / 1e6
        print(f"GPU memory in use before measuring: {used_mb:.0f} MB")
        if used_mb > 4000:
            print("WARNING: the GPU appears busy; latency will be inflated.")

    rows = {}
    for name in ["MyoDigits-TX", "MyoDigits-GRU", "MyoDigits-PW"]:
        m = build(name).eval()
        params = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        x = torch.randn(1, SEQ, IN_CH, WIN)

        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc:
                m(x)
            flops = fc.get_total_flops()          # counts 2 per MAC
        macs = flops / 2.0

        gpu_med = gpu_p95 = peak_mb = float("nan")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gpu_med, gpu_p95 = bench(m, x, "cuda")
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            m = m.cpu()
        cpu_med, cpu_p95 = bench(m, x, "cpu", n=40, warmup=10)

        rows[name] = dict(
            params=params, trainable=trainable,
            macs=macs, gmacs=macs / 1e9,
            size_mb=params * 4 / 1e6,
            peak_mem_mb=peak_mb,
            gpu_ms=gpu_med, gpu_p95_ms=gpu_p95,
            cpu_ms=cpu_med, cpu_p95_ms=cpu_p95,
            realtime_factor=(SEQ * 0.05 * 1000.0) / gpu_med if gpu_med == gpu_med else None,
        )
        print(f"{name:16s} {params:>9,} params  {macs/1e9:7.3f} GMAC  "
              f"gpu {gpu_med:6.2f} ms  cpu {cpu_med:7.2f} ms  "
              f"peak {peak_mb:6.1f} MB")
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    meta = dict(seq_len=SEQ, in_ch=IN_CH, win_samples=WIN,
                context_seconds=SEQ * 0.05,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                torch=torch.__version__)
    json.dump(dict(meta=meta, models=rows), open(OUT, "w"), indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
