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
"""Oracle calibration ceiling on the CORRECTED v4 headline (run28).
Replaces the stale run26/loso_v2 oracle numbers in Section 5.7."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, str(_ROOT / "src"))
import torch
from torch.utils.data import DataLoader

ROOT = str(_ROOT)
RUN  = f"{ROOT}/experiments/run28_posenet_tx_lr3e-4_bs64_ep60_loso_v4headline"
DATA = f"{ROOT}/data/processed/myodigitsv1_loso_v4"
OUTD = str(_THESIS / "figures")
CACHE = f"{OUTD}/oracle_v4_cache"
os.makedirs(CACHE, exist_ok=True)

from myodigits.models.posenet_tx import PoseNetTx
from myodigits.data.dataset import SeqDataset
meta = json.load(open(f"{DATA}/meta.json"))["per_user"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

summ = json.load(open(f"{RUN}/loso_summary.json"))
users = sorted(f.get("test_user", f["fold"]) for f in summ["per_fold"])

# ---------- 1. predictions for every held-out subject ----------
for u in users:
    out = f"{CACHE}/U{u:02d}.npz"
    if os.path.exists(out):
        continue
    fd = glob.glob(f"{RUN}/fold*_testUser{u:02d}")
    if not fd: print(f"  no fold for U{u:02d}"); continue
    ck = f"{fd[0]}/checkpoints/best.pt"
    m = PoseNetTx(in_ch=16, out_dim=20, base=48, tx_heads=4, tx_layers=2, tx_dim_ff=256)
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    m.load_state_dict(raw["model_state_dict"], strict=True)
    m.to(dev).eval()
    ds = SeqDataset(f"{DATA}/User{u}", seq_len=64, verbose=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    P, T = [], []
    with torch.no_grad():
        for xb, yb in dl:
            p = m(xb.to(dev)).cpu().numpy()
            P.append(p.reshape(-1, p.shape[-1])); T.append(yb.numpy().reshape(-1, yb.shape[-1]))
    P = np.concatenate(P); T = np.concatenate(T)
    st = meta[f"User{u}"]
    mu = np.array(st["jnt_mean"], np.float32); sd = np.array(st["jnt_std"], np.float32)
    P = P*sd + mu; T = T*sd + mu
    np.savez_compressed(out, pred=P.astype(np.float32), targ=T.astype(np.float32))
    print(f"  U{u:02d} preds  MAE {np.abs(P-T).mean():.3f}", flush=True)

# ---------- 2. oracle ladder ----------
from sklearn.isotonic import IsotonicRegression

def mae(a, b): return float(np.abs(a-b).mean())

def affine_ls(P, T):
    """per-joint least-squares gain+bias fitted on the test labels (oracle)."""
    O = np.empty_like(P)
    for j in range(P.shape[1]):
        x, y = P[:, j], T[:, j]
        A = np.vstack([x, np.ones_like(x)]).T
        g, b = np.linalg.lstsq(A, y, rcond=None)[0]
        O[:, j] = g*x + b
    return O

def affine_mae_opt(P, T, grid=41):
    """per-joint gain grid + median-optimal bias (minimizes MAE, not MSE)."""
    O = np.empty_like(P)
    for j in range(P.shape[1]):
        x, y = P[:, j], T[:, j]
        sx = x.std() or 1.0
        gains = np.linspace(0.2, 3.0, grid) * (y.std()/sx if sx else 1.0)
        best, bg, bb = None, 1.0, 0.0
        for g in gains:
            b = np.median(y - g*x)
            e = np.abs(g*x + b - y).mean()
            if best is None or e < best: best, bg, bb = e, g, b
        O[:, j] = bg*x + bb
    return O

def isotonic(P, T, sub=60000):
    O = np.empty_like(P)
    for j in range(P.shape[1]):
        x, y = P[:, j], T[:, j]
        idx = np.random.RandomState(0).choice(len(x), min(sub, len(x)), replace=False)
        ir = IsotonicRegression(out_of_bounds="clip").fit(x[idx], y[idx])
        O[:, j] = ir.predict(x)
    return O

rows = {}
for u in users:
    f = f"{CACHE}/U{u:02d}.npz"
    if not os.path.exists(f): continue
    z = np.load(f); P, T = z["pred"].astype(np.float64), z["targ"].astype(np.float64)
    rows[u] = {"baseline": mae(P, T),
               "affine_ls": mae(affine_ls(P, T), T),
               "affine_mae": mae(affine_mae_opt(P, T), T),
               "isotonic":  mae(isotonic(P, T), T)}
    print(f"  U{u:02d} oracle: " + "  ".join(f"{k}={v:.3f}" for k, v in rows[u].items()), flush=True)

res = {}
for k in ["baseline", "affine_ls", "affine_mae", "isotonic"]:
    v = np.array([rows[u][k] for u in sorted(rows)])
    res[k] = {"mean": float(v.mean()), "best": float(v.min()), "worst": float(v.max()),
              "spread": float(v.max()/v.min())}
res["per_user"] = {str(u): rows[u] for u in sorted(rows)}
json.dump(res, open(f"{OUTD}/oracle_v4.json", "w"), indent=2)

print("\n=== ORACLE CEILING ON CORRECTED v4 (run28) ===")
print(f"{'condition':<14}{'mean':>8}{'best':>8}{'worst':>8}{'spread':>9}")
for k in ["baseline", "affine_ls", "affine_mae", "isotonic"]:
    r = res[k]
    print(f"{k:<14}{r['mean']:8.3f}{r['best']:8.3f}{r['worst']:8.3f}{r['spread']:8.2f}x")
