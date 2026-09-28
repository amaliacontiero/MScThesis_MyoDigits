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
"""Deployment cost of the three architectures.

Three questions the whole-sequence timing does not answer:

  1. What does it cost to emit ONE new prediction when a new window arrives?
     A deployed system runs at the 50 ms hop, not once per 3.2 s sequence.
     A causal decoder can advance one step; a bidirectional one must recompute
     the whole window, so the streaming cost differs from the offline cost.
  2. What does eight-bit dynamic quantisation buy?
  3. How much of the 50 ms budget does each option consume?

Latency must be measured on an idle device. A measurement taken while the same
GPU is training is inflated several-fold and is not reported here.

Writes figures2/deployment.json.
"""
import json, os, sys, time, copy, tempfile
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(_ROOT / "src"))
OUT = str(_THESIS / "figures2/deployment.json")

SEQ, IN_CH, WIN, HOP_MS = 64, 16, 400, 50.0


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


def bench(fn, n=200, warmup=30, cuda=False):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def disk_mb(m):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(m.state_dict(), f.name)
        mb = os.path.getsize(f.name) / 1e6
    os.unlink(f.name)
    return mb


def main():
    cuda = torch.cuda.is_available()
    if cuda:
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1e6
        print(f"GPU in use before measuring: {used:.0f} MB"
              + ("   WARNING: busy, latency will be inflated" if used > 4000 else ""))

    rows = {}
    for name in ["MyoDigits-TX", "MyoDigits-GRU", "MyoDigits-PW"]:
        m = build(name).eval()
        seq = torch.randn(1, SEQ, IN_CH, WIN)
        one = torch.randn(1, 1, IN_CH, WIN)
        r = {"params": sum(p.numel() for p in m.parameters()),
             "fp32_mb": disk_mb(m)}

        with torch.no_grad():
            # offline: one pass over the whole 3.2 s sequence
            if cuda:
                mc, xc = m.cuda(), seq.cuda()
                r["gpu_seq_ms"], _ = bench(lambda: mc(xc), cuda=True)
                # streaming, naive: recompute the whole window every hop
                r["gpu_hop_recompute_ms"] = r["gpu_seq_ms"]
                m = m.cpu()
            r["cpu_seq_ms"], _ = bench(lambda: m(seq), n=40, warmup=8)
            r["cpu_hop_recompute_ms"] = r["cpu_seq_ms"]
            # cost of encoding a single newly arrived window
            r["cpu_one_window_ms"], _ = bench(lambda: m(one), n=60, warmup=10)

            # eight-bit dynamic quantisation (CPU; covers Linear and RNN layers)
            try:
                qm = torch.quantization.quantize_dynamic(
                    copy.deepcopy(m), {nn.Linear, nn.GRU, nn.LSTM}, dtype=torch.qint8).eval()
                r["int8_mb"] = disk_mb(qm)
                r["cpu_seq_int8_ms"], _ = bench(lambda: qm(seq), n=40, warmup=8)
            except Exception as e:
                r["int8_error"] = str(e)[:120]

        r["hop_budget_ms"] = HOP_MS
        r["gpu_hop_headroom"] = HOP_MS / r["gpu_hop_recompute_ms"] if cuda else None
        r["cpu_hop_headroom"] = HOP_MS / r["cpu_hop_recompute_ms"]
        rows[name] = r
        print(f"\n{name}")
        print(f"  params {r['params']:>9,}   fp32 {r['fp32_mb']:.2f} MB"
              + (f"   int8 {r['int8_mb']:.2f} MB" if "int8_mb" in r else ""))
        if cuda:
            print(f"  GPU  sequence {r['gpu_seq_ms']:6.2f} ms   "
                  f"headroom vs 50 ms hop {r['gpu_hop_headroom']:5.1f}x")
        print(f"  CPU  sequence {r['cpu_seq_ms']:6.2f} ms   "
              f"headroom {r['cpu_hop_headroom']:5.1f}x")
        print(f"  CPU  single window {r['cpu_one_window_ms']:6.2f} ms")
        if "cpu_seq_int8_ms" in r:
            print(f"  CPU  sequence int8 {r['cpu_seq_int8_ms']:6.2f} ms "
                  f"({r['cpu_seq_ms']/r['cpu_seq_int8_ms']:.2f}x)")
        del m

    meta = dict(seq_len=SEQ, hop_ms=HOP_MS, window_ms=200,
                device=torch.cuda.get_device_name(0) if cuda else "cpu",
                torch=torch.__version__)
    json.dump(dict(meta=meta, models=rows), open(OUT, "w"), indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
