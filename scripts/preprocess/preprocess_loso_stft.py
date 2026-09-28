"""
LOSO preprocessing for the PiMForce STFT model (EMGNetAngles).

Faithful to PiMForce's EMG representation (Seo et al., NeurIPS 2024):
  - RAW EMG, 8 channels, NO bandpass / notch / RMS
    (the STFT magnitude spectrogram is computed inside the model)

Adapted for our cross-subject LOSO protocol:
  - per-user z-score of EMG  (DEVIATES from their raw EMG; needed because per-user
    amplitude differences otherwise dominate the spectrogram across subjects)
  - per-user z-score of joint angles (denorm stats stored for eval -> degrees)
  - data organised per-user (UserN/) like loso_v2/v3, not their session split

Stores CONTINUOUS per-trial arrays (not pre-windowed) — the STFT model's 1248-sample
windows overlap heavily, so the dataset builds the window index on the fly.

Output: data/processed/myodigitsv1_loso_stft/UserN/seg_XXXX.npz
        X shape: (T_trial, 8)    z-scored raw EMG
        y shape: (T_trial, 20)   z-scored joint angles
        meta.json holds per-user emg_mean/std and jnt_mean/std.

Run from project root:
    python scripts/preprocess/preprocess_loso_stft.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import os
ROOT      = Path(__file__).resolve().parents[2]          # repository root
DATA_ROOT = Path(os.environ.get("MYODIGITS_RAW_DATA", ROOT / "data/raw/pimforce"))
OUT_ROOT  = ROOT / "data/processed/myodigitsv1_loso_stft"

FS       = 2000
N_USERS  = 21
SESSIONS = [1, 2, 3]


def _load_trial_csvs(user_dir):
    """Yield (raw_emg (T,8), raw_joints (T,20), filename) for every trial of a user."""
    emg_cols = jnt_cols = None
    for sess in SESSIONS:
        sess_dir = user_dir / f"Session{sess}"
        if not sess_dir.exists():
            continue
        for csv_path in sorted(sess_dir.glob("join_*.csv"),
                               key=lambda p: int(p.stem.split("_")[1])):
            df = pd.read_csv(csv_path)
            if emg_cols is None:
                emg_cols = [c for c in df.columns if "sEMG" in c]
                jnt_cols = [c for c in df.columns if c.startswith("[")]
                if len(emg_cols) != 8:
                    raise ValueError(f"Expected 8 EMG cols, got {len(emg_cols)}")
                if len(jnt_cols) != 20:
                    raise ValueError(f"Expected 20 joint cols, got {len(jnt_cols)}")
            yield (df[emg_cols].to_numpy(dtype=np.float64),
                   df[jnt_cols].to_numpy(dtype=np.float64),
                   csv_path.name)


def process_user(user_id, out_dir):
    user_dir = DATA_ROOT / f"User{user_id}"

    trials = list(_load_trial_csvs(user_dir))
    if not trials:
        print(f"  User{user_id:02d}: no data found, skipping")
        return None

    concat_emg = np.concatenate([t[0] for t in trials], axis=0)
    concat_jnt = np.concatenate([t[1] for t in trials], axis=0)

    emg_mean = concat_emg.mean(0); emg_std = np.maximum(concat_emg.std(0), 1e-8)
    jnt_mean = concat_jnt.mean(0); jnt_std = np.maximum(concat_jnt.std(0), 1e-8)

    seg_idx     = 0
    total_steps = 0
    for raw_emg, raw_jnts, _ in trials:
        emg_norm = ((raw_emg  - emg_mean) / emg_std).astype(np.float32)   # (T, 8)
        jnt_norm = ((raw_jnts - jnt_mean) / jnt_std).astype(np.float32)   # (T, 20)
        np.savez(out_dir / f"seg_{seg_idx:04d}.npz", X=emg_norm, y=jnt_norm)
        seg_idx     += 1
        total_steps += emg_norm.shape[0]

    return {
        "samples":  total_steps,
        "segments": seg_idx,
        "emg_mean": emg_mean.tolist(), "emg_std": emg_std.tolist(),
        "jnt_mean": jnt_mean.tolist(), "jnt_std": jnt_std.tolist(),
    }


def main():
    print("=" * 60)
    print("MyoDigits LOSO Preprocessor — STFT (raw EMG, per-user z-score)")
    print("=" * 60)
    print(f"  Source: {DATA_ROOT}")
    print(f"  Output: {OUT_ROOT}\n")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0       = time.time()
    all_meta = {}

    for user_id in range(1, N_USERS + 1):
        out_dir = OUT_ROOT / f"User{user_id}"
        out_dir.mkdir(exist_ok=True)
        stats = process_user(user_id, out_dir)
        if stats is None:
            continue
        all_meta[f"User{user_id}"] = stats
        print(f"  User{user_id:02d}: {stats['samples']:,} samples in {stats['segments']} segments")

    meta = {
        "version":  "loso_stft",
        "pipeline": "preprocess_loso_stft.py",
        "preprocessing": {
            "emg":        "raw (no filter/notch/RMS), per-user zscore",
            "n_channels": 8,
            "joint_norm": "per_user_zscore",
            "note":       "STFT spectrogram computed inside the model (EMGNetAngles)",
        },
        "stft": {
            "window_length":   256,
            "hop_length":      32,
            "num_frames":      32,
            "num_frequencies": 64,
            "input_len":       (32 + 256 // 32 - 1) * 32,   # 1248
            "fs":              FS,
        },
        "per_user": all_meta,
    }
    (OUT_ROOT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"Saved meta.json -> {OUT_ROOT / 'meta.json'}")


if __name__ == "__main__":
    main()
