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
"""Regenerate v4 predictions from run28 checkpoints for qualitative figures.
Uses the CORRECTED loso_v4 data (not the stale oracle_cache from run26/v2)."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, str(_ROOT / "src"))
import torch
from torch.utils.data import DataLoader

ROOT = str(_ROOT)
RUN  = f"{ROOT}/experiments/run28_posenet_tx_lr3e-4_bs64_ep60_loso_v4headline"
DATA = f"{ROOT}/data/processed/myodigitsv1_loso_v4"
OUT  = str(_THESIS / "figures/preds")
os.makedirs(OUT, exist_ok=True)

from myodigits.models.posenet_tx import PoseNetTx
from myodigits.data.dataset import SeqDataset   # noqa
stats = json.load(open(f"{DATA}/meta.json"))["per_user"]

# best / median / worst by run28 per-fold MAE
TARGETS = {5: "best", 13: "median", 12: "worst"}
dev = "cuda" if torch.cuda.is_available() else "cpu"

for user, label in TARGETS.items():
    fold_dirs = glob.glob(f"{RUN}/fold*_testUser{user:02d}")
    if not fold_dirs:
        print(f"no fold dir for U{user:02d}"); continue
    ck = f"{fold_dirs[0]}/checkpoints/best.pt"
    if not os.path.exists(ck):
        print("no checkpoint", ck); continue
    model = PoseNetTx(in_ch=16, out_dim=20, base=48, tx_heads=4, tx_layers=2, tx_dim_ff=256)
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"] if "model_state_dict" in raw else raw.get("state_dict", raw)
    missing, unexpected = model.load_state_dict(sd, strict=True)   # strict: fail loudly
    model.to(dev).eval()

    # non-overlapping sequences, matching the evaluation protocol
    ds = SeqDataset(f"{DATA}/User{user}", seq_len=64, verbose=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    P, T = [], []
    with torch.no_grad():
        for xb, yb in dl:                      # FULL test set, no early break
            p = model(xb.to(dev)).cpu().numpy()
            P.append(p.reshape(-1, p.shape[-1])); T.append(yb.numpy().reshape(-1, yb.shape[-1]))
    P = np.concatenate(P); T = np.concatenate(T)

    st = stats.get(f"User{user}", {})
    if "jnt_mean" in st:
        m = np.array(st["jnt_mean"], dtype=np.float32); s = np.array(st["jnt_std"], dtype=np.float32)
        P = P*s + m; T = T*s + m
    np.savez_compressed(f"{OUT}/U{user:02d}_{label}.npz", pred=P, targ=T)
    mae = np.abs(P-T).mean()
    # trivial baseline: predict this user's mean pose for every window
    base = np.abs(T - T.mean(0, keepdims=True)).mean()
    REPORTED = {5: 3.97, 13: 6.96, 12: 9.89}
    print(f"U{user:02d} ({label:6s}): MAE {mae:5.2f} deg  (run28 reported {REPORTED[user]:.2f})"
          f"   mean-pose baseline {base:5.2f} deg   n={P.shape[0]}")
print("done")
