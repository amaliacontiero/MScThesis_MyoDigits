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
"""Predict a CONTINUOUS timeline for single trials.

The evaluation dataset builds sequences with stride 1, so flattening its output
does not give a timeline. Here each trial is cut into NON-OVERLAPPING 64-window
sequences, so concatenating the predictions yields genuine continuous time at
one window per 50 ms hop, entirely within one recording.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, str(_ROOT / "src"))
import torch
from myodigits.models.posenet_tx import PoseNetTx

ROOT = str(_ROOT)
RUN  = f"{ROOT}/experiments/run28_posenet_tx_lr3e-4_bs64_ep60_loso_v4headline"
DATA = f"{ROOT}/data/processed/myodigitsv1_loso_v4"
OUT  = str(_THESIS / "figures/trials")
os.makedirs(OUT, exist_ok=True)
SEQ, HOP_S = 64, 0.05          # 64 windows per sequence; 100-sample hop at 2 kHz

def predict_trials(user, max_trials=None):
    fd = glob.glob(f"{RUN}/fold*_testUser{user:02d}")
    assert fd, f"no fold for U{user:02d}"
    m = PoseNetTx(in_ch=16, out_dim=20, base=48, tx_heads=4, tx_layers=2, tx_dim_ff=256)
    raw = torch.load(f"{fd[0]}/checkpoints/best.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(raw["model_state_dict"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev).eval()

    st = json.load(open(f"{DATA}/meta.json"))["per_user"][f"User{user}"]
    mu = np.array(st["jnt_mean"], np.float32); sd = np.array(st["jnt_std"], np.float32)

    out = []
    segs = sorted(glob.glob(f"{DATA}/User{user}/seg_*.npz"))
    if max_trials: segs = segs[:max_trials]
    for si, sp in enumerate(segs):
        d = np.load(sp)
        X = torch.from_numpy(d["X"]).float()      # (n_windows, 16, 400)
        Y = d["y"]                                # (n_windows, 20)
        n = (len(X) // SEQ) * SEQ                 # whole sequences only
        if n == 0: continue
        xb = X[:n].reshape(-1, SEQ, X.shape[1], X.shape[2])
        preds = []
        with torch.no_grad():
            for i in range(0, len(xb), 16):
                p = m(xb[i:i+16].to(dev)).cpu().numpy()
                preds.append(p.reshape(-1, p.shape[-1]))
        P = np.concatenate(preds)                 # (n, 20) CONTINUOUS in time
        T = Y[:n]
        out.append(dict(trial=os.path.basename(sp), pred=P*sd+mu, targ=T*sd+mu))
    return out

if __name__ == "__main__":
    user = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    trials = predict_trials(user)
    print(f"U{user:02d}: {len(trials)} trials")
    # sanity: within a trial the timeline must be smooth (no 64-frame sawtooth)
    best = None
    for tr in trials:
        T, P = tr["targ"], tr["pred"]
        d = np.abs(np.diff(T, axis=0)).mean(axis=1)
        jumps = int((d > 15).sum())
        move = T.std(axis=0).mean()
        err  = np.abs(T - P).mean()
        tr.update(jumps=jumps, move=move, err=err, n=len(T))
        if move > 6 and (best is None or move/err > best["move"]/best["err"]):
            best = tr
    tot_jumps = sum(t["jumps"] for t in trials)
    print(f"  discontinuities >15 deg inside trials: {tot_jumps} "
          f"(was every 64 frames before the fix)")
    print(f"  best trial: {best['trial']}  n={best['n']}  "
          f"movement {best['move']:.1f} deg  error {best['err']:.2f} deg  "
          f"duration {best['n']*HOP_S:.1f}s")
    np.savez_compressed(f"{OUT}/U{user:02d}_best_trial.npz",
                        pred=best["pred"], targ=best["targ"], trial=best["trial"])
    # also keep a few long ones for the thesis figure
    longest = sorted(trials, key=lambda t: -t["n"])[:3]
    np.savez_compressed(f"{OUT}/U{user:02d}_trials.npz",
                        **{f"pred{i}": t["pred"] for i, t in enumerate(longest)},
                        **{f"targ{i}": t["targ"] for i, t in enumerate(longest)})
    print("  saved to", OUT)
