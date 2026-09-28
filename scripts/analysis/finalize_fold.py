"""
Finalize a fold that was interrupted mid-training: run eval on its saved
best.pt (the best-val checkpoint) so metrics.json / plots / results are written
and the fold counts as complete (skip-guard will then skip it on resume).

Usage:
  python scripts/finalize_fold.py <run_dir> <test_user> \
         --model posenet_tx --in_ch 8 --base 48 --data loso_v3
"""
import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import torch
from torch.utils.data import DataLoader

from myodigits.data.dataset import SeqDataset
from myodigits.training.eval import run_eval

MODEL_REGISTRY = {
    "posenet_tx":      ("myodigits.models.posenet_tx",      "PoseNetTx"),
    "posenet_tx_stft": ("myodigits.models.posenet_tx_stft", "PoseNetTxStft"),
    "posenet_seq":     ("myodigits.models.posenet_seq",     "PoseNetSeq"),
}


def build_model(name, in_ch, base):
    import importlib
    mod, cls = MODEL_REGISTRY[name]
    M = getattr(importlib.import_module(mod), cls)
    return M(in_ch=in_ch, base=base, tx_heads=4, tx_layers=2, tx_dim_ff=256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("test_user", type=int)
    ap.add_argument("--model", default="posenet_tx")
    ap.add_argument("--in_ch", type=int, default=8)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--data", default="myodigitsv1_loso_v3")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--n_users", type=int, default=21)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    run_dir = (root / "experiments" / args.run_dir) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    data_root = root / "data/processed" / args.data

    all_users = list(range(1, args.n_users + 1))
    fold_idx = all_users.index(args.test_user)
    val_users = [all_users[(fold_idx + 1 + k) % args.n_users] for k in range(3)]
    fold_dir = run_dir / f"fold{fold_idx+1:02d}_testUser{args.test_user:02d}"
    ckpt = fold_dir / "checkpoints" / "best.pt"
    assert ckpt.exists(), f"no best.pt at {ckpt}"
    if (fold_dir / "metrics.json").exists():
        print(f"metrics.json already exists at {fold_dir} — nothing to do.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.in_ch, args.base).to(device)

    ds_test = SeqDataset(data_root / f"User{args.test_user}", seq_len=args.seq_len, verbose=False)
    loader_test = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    hist_path = fold_dir / "checkpoints" / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else {}

    meta_path = data_root / "meta.json"
    per_user = json.loads(meta_path.read_text()).get("per_user", {}) if meta_path.exists() else {}
    us = per_user.get(f"User{args.test_user}", {})
    jnt_denorm = (np.array(us["jnt_mean"], np.float32), np.array(us["jnt_std"], np.float32)) if "jnt_mean" in us else None

    fold_config = {"smooth_lambda": 0.0, "batch_size": 64,
                   "checkpoint_dir": str(fold_dir / "checkpoints")}
    fold_label = f"{run_dir.name}_fold{fold_idx+1:02d}_testUser{args.test_user:02d}"

    metrics = run_eval(model, loader_test, history, fold_config, device,
                       fold_dir, fold_label, jnt_denorm=jnt_denorm)
    # match launch.py's extra fields so the summary stays consistent
    metrics["fold"], metrics["test_user"], metrics["val_users"] = fold_idx + 1, args.test_user, val_users
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nFinalized fold{fold_idx+1:02d} (User{args.test_user}) from best.pt: "
          f"test_mae={metrics['test_mae']:.4f}")


if __name__ == "__main__":
    main()
