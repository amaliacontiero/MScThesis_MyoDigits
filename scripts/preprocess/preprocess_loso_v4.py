"""
LOSO preprocessing v4 = CORRECTED headline pipeline.

Same as v2 (16 channels: 8 filtered + 8 RMS) but with the notch fixed to 60 Hz
(PiMForce = KAIST, South Korea, 60 Hz grid). v2 used a wrong 50 Hz notch.

Signal chain:
  raw sEMG (8ch)
    -> zero-phase bandpass 20-450 Hz (sosfiltfilt, order 4)
    -> zero-phase 60 Hz notch (filtfilt, Q=30)      [8 filtered channels]
    -> RMS envelope, 50 ms symmetric window          [8 RMS channels]
    -> stack -> 16 channels
    -> per-user z-score (16ch)
  joints -> per-user z-score

Output: data/processed/myodigitsv1_loso_v4/UserN/seg_XXXX.npz
        X (N, 16, 400), y (N, 20)

Recipe values taken from loso_v2/meta.json (bandpass, order, notch_q, rms_window_ms,
rms_type) with notch_hz corrected 50 -> 60.
Run: python scripts/preprocess/preprocess_loso_v4.py
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

import os
ROOT      = Path(__file__).resolve().parents[2]          # repository root
DATA_ROOT = Path(os.environ.get("MYODIGITS_RAW_DATA", ROOT / "data/raw/pimforce"))
OUT_ROOT  = Path(os.environ.get("MYODIGITS_DATA_ROOT", ROOT / "data/processed/myodigitsv1_loso_v4"))

FS          = 2000
WIN_SAMPLES = 400
HOP_SAMPLES = 100
N_USERS     = 21
SESSIONS    = [1, 2, 3]

BP_LO, BP_HI, BP_ORDER = 20.0, 450.0, 4
NOTCH_F, NOTCH_Q = 60.0, 30.0          # corrected: 60 Hz (Korea)
RMS_WIN_MS = 50.0
RMS_WIN = int(round(RMS_WIN_MS / 1000.0 * FS))   # 100 samples


def _design():
    sos_bp = butter(BP_ORDER, [BP_LO, BP_HI], btype="band", fs=FS, output="sos")
    b_n, a_n = iirnotch(NOTCH_F, NOTCH_Q, FS)
    return sos_bp, (b_n, a_n)


def _filter(raw, sos_bp, notch):
    out = np.empty_like(raw, dtype=np.float64)
    for ch in range(raw.shape[1]):
        x = sosfiltfilt(sos_bp, raw[:, ch].astype(np.float64))
        out[:, ch] = filtfilt(notch[0], notch[1], x)
    return out


def _rms(sig, win):
    """Symmetric (centered) moving RMS of each channel."""
    k = np.ones(win) / win
    out = np.empty_like(sig)
    for ch in range(sig.shape[1]):
        out[:, ch] = np.sqrt(np.convolve(sig[:, ch] ** 2, k, mode="same"))
    return out


def _load_trials(user_dir):
    emg_cols = jnt_cols = None
    for sess in SESSIONS:
        sd = user_dir / f"Session{sess}"
        if not sd.exists():
            continue
        for csv in sorted(sd.glob("join_*.csv"), key=lambda p: int(p.stem.split("_")[1])):
            df = pd.read_csv(csv)
            if emg_cols is None:
                emg_cols = [c for c in df.columns if "sEMG" in c]
                jnt_cols = [c for c in df.columns if c.startswith("[")]
                assert len(emg_cols) == 8 and len(jnt_cols) == 20
            yield df[emg_cols].to_numpy(np.float64), df[jnt_cols].to_numpy(np.float64), csv.name


def process_user(uid, filt, out_dir):
    trials = []
    all_emg, all_jnt = [], []
    for raw_emg, raw_jnt, fname in _load_trials(DATA_ROOT / f"User{uid}"):
        filt8 = _filter(raw_emg, filt[0], filt[1])          # (T,8)
        rms8 = _rms(filt8, RMS_WIN)                          # (T,8)
        emg16 = np.concatenate([filt8, rms8], axis=1)       # (T,16)
        trials.append((emg16, raw_jnt, fname))
        all_emg.append(emg16); all_jnt.append(raw_jnt)
    if not trials:
        return None
    ce = np.concatenate(all_emg); cj = np.concatenate(all_jnt)
    emg_mean, emg_std = ce.mean(0), np.maximum(ce.std(0), 1e-8)
    jnt_mean, jnt_std = cj.mean(0), np.maximum(cj.std(0), 1e-8)

    seg, tot = 0, 0
    for emg16, raw_jnt, _ in trials:
        en = ((emg16 - emg_mean) / emg_std).astype(np.float32)
        jn = ((raw_jnt - jnt_mean) / jnt_std).astype(np.float32)
        Xs, ys = [], []
        for s in range(0, len(en) - WIN_SAMPLES + 1, HOP_SAMPLES):
            Xs.append(en[s:s + WIN_SAMPLES].T)     # (16,400)
            ys.append(jn[s + WIN_SAMPLES - 1])     # (20,)
        if Xs:
            np.savez(out_dir / f"seg_{seg:04d}.npz", X=np.stack(Xs), y=np.stack(ys))
            seg += 1; tot += len(Xs)
    return tot, seg, {"emg_mean": emg_mean.tolist(), "emg_std": emg_std.tolist(),
                      "jnt_mean": jnt_mean.tolist(), "jnt_std": jnt_std.tolist()}


def main():
    print("MyoDigits LOSO Preprocessor v4  (16ch, 60Hz notch CORRECTED)")
    print(f"  RMS window = {RMS_WIN} samples ({RMS_WIN_MS} ms)")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    filt = _design(); t0 = time.time(); all_meta = {}
    for uid in range(1, N_USERS + 1):
        od = OUT_ROOT / f"User{uid}"; od.mkdir(exist_ok=True)
        res = process_user(uid, filt, od)
        if res is None:
            print(f"  User{uid:02d}: no data"); continue
        tot, seg, stats = res
        all_meta[f"User{uid}"] = {"windows": tot, "segments": seg, **stats}
        print(f"  User{uid:02d}: {tot:,} windows, {seg} segs", flush=True)
    meta = {
        "version": "loso_v4", "pipeline": "preprocess_loso_v4.py",
        "preprocessing": {"bandpass_hz": [BP_LO, BP_HI], "bandpass_order": BP_ORDER,
                          "filter_type": "zero_phase_sosfiltfilt", "notch_hz": NOTCH_F,
                          "notch_q": NOTCH_Q, "rms_window_ms": RMS_WIN_MS,
                          "rms_type": "symmetric_convolution", "n_channels": 16,
                          "emg_norm": "per_user_zscore", "joint_norm": "per_user_zscore",
                          "note": "corrected 60Hz notch (Korea); recipe matches v2 except notch"},
        "windowing": {"win_samples": WIN_SAMPLES, "hop_samples": HOP_SAMPLES, "fs": FS},
        "per_user": all_meta,
    }
    (OUT_ROOT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {time.time()-t0:.0f}s -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
