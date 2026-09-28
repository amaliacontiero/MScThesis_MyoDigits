"""
Re-run eval (metrics + plots + animation) for a single LOSO fold.

Usage:
    python scripts/eval_fold.py <fold_dir> <test_user_id>

Example:
    python scripts/eval_fold.py \
        experiments/run06_posenet_seq_lr3e-4_bs64_ep60_loso/fold01_testUser01 1
"""

import sys
import json
import importlib
import inspect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from myodigits.data.dataset import SeqDataset
from myodigits.training.eval import run_eval

MODEL_REGISTRY = {
    "cnn_v1":       ("myodigits.models.cnn_v1",       "CNN"),
    "cnn_v2":       ("myodigits.models.cnn_v2",       "CNN"),
    "lstm_v1":      ("myodigits.models.lstm_v1",      "LSTM"),
    "posenet_seq":  ("myodigits.models.posenet_seq",  "PoseNetSeq"),
}

# ── config matching the original run ─────────────────────────────────────────
CONFIG = {
    "model":          "posenet_seq",
    "batch_size":     64,
    "loso_data_root": "data/processed/myodigitsv1_loso_v2",
    "in_ch":          16,
    "seq_len":        64,
}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    fold_dir    = ROOT / sys.argv[1]
    test_user   = int(sys.argv[2])
    ckpt_dir    = fold_dir / "checkpoints"
    history_path = ckpt_dir / "history.json"

    history = json.loads(history_path.read_text())

    module_path, class_name = MODEL_REGISTRY[CONFIG["model"]]
    ModelClass = getattr(importlib.import_module(module_path), class_name)

    sig = inspect.signature(ModelClass.__init__)
    kwargs = {}
    if "in_ch" in sig.parameters:
        kwargs["in_ch"] = CONFIG["in_ch"]
    model = ModelClass(**kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loso_data_root = ROOT / CONFIG["loso_data_root"]
    ds_test = SeqDataset(loso_data_root / f"User{test_user}",
                         seq_len=CONFIG["seq_len"], verbose=True)
    loader_test = DataLoader(ds_test, batch_size=CONFIG["batch_size"], shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    fold_config = {"checkpoint_dir": str(ckpt_dir)}

    # load per-user joint denorm stats if available (v2 data has normalized joints)
    meta_path = loso_data_root / "meta.json"
    jnt_denorm = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        user_stats = meta.get("per_user", {}).get(f"User{test_user}")
        if user_stats and "jnt_mean" in user_stats:
            jnt_denorm = (
                np.array(user_stats["jnt_mean"], dtype=np.float32),
                np.array(user_stats["jnt_std"],  dtype=np.float32),
            )
            print(f"Loaded joint denorm stats for User{test_user}")

    run_name = fold_dir.parent.name + "_" + fold_dir.name
    run_eval(model, loader_test, history, fold_config, device, fold_dir, run_name,
             jnt_denorm=jnt_denorm)


if __name__ == "__main__":
    main()
