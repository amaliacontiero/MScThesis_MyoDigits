import gc
import sys
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import os as _os
STRICT_NORM = _os.environ.get("MYODIGITS_STRICT_NORM", "0") == "1"
if STRICT_NORM:
    sys.path.insert(0, str(ROOT / "scripts" / "normalisation"))
    from strict_norm import load_meta as _sn_load_meta, fold_statistics as _sn_fold_stats, save_fold_artifact as _sn_save
    from strict_datasets import StrictLosoSeqDataset, StrictSeqDataset
    print("[STRICT NORM] Experiment A active: statistics fitted on TRAINING participants only")

CONFIG = {
    # ── architecture ──────────────────────────────────────────────────────────
    # run11 (medium Tx):              posenet_tx         | base=64  | loso_v2 | 16ch
    # run13 (large Tx):               posenet_tx         | base=128 | loso_v2 | 16ch
    # run14 (small Tx, baseline):     posenet_tx         | base=48  | loso_v2 | 16ch
    # run15 (small Tx, noBN+noResid): posenet_tx_v2      | base=64  | loso_v2 | 16ch
    # run16 (small Tx, noBN only):    posenet_tx_noBN    | base=48  | loso_v2 | 16ch
    # run17 (small Tx, noResid only): posenet_tx_noResid | base=48  | loso_v2 | 16ch
    # run18 (small Tx, v3 data):      posenet_tx         | base=48  | loso_v3 | 8ch  ← current
    # run24 (PiMForce port):          emgnet_stft        | raw 8ch  | STFT 2D-CNN  ← current
    # ── SMALL PoseNetTx — FULL 21-fold LOSO (best-model attempt vs run06) ───────
    # Strongest known small-Tx recipe (= run23 strongAug, 6.87 proxy): base=48,
    # 16ch v2, lr3e-4, wd1e-2, AdamW, augmentation ON (targets cross-subject gap).
    # First full-LOSO number for the PoseNetTx family; challenges run06 (7.28).
    "model":          "posenet_tx",   # run26 headline, re-run on CORRECTED v4 data
    "in_ch":          16,             # 16ch (8 filtered + 8 RMS), loso_v4
    "model_kwargs":   {"base": 48, "tx_heads": 4, "tx_layers": 2, "tx_dim_ff": 256},
    "tag":            "v4headline",
    "stft":           False,           # use standard seq pipeline; stem does STFT internally
    "stft_win_hop":   256,             # (unused when stft=False)

    # ── training ──────────────────────────────────────────────────────────────
    # Recipe R — identical to E1/run25 for the controlled front-end comparison.
    "epochs":         60,
    "lr":             3e-4,
    "batch_size":     64,
    "patience":       20,
    "weight_decay":   1e-2,
    "optimizer":      "adamw",
    "smooth_lambda":  0.0,      # 0.0 = disabled
    "augment":        True,     # run26 recipe: sEMG augmentation ON

    # ── data ──────────────────────────────────────────────────────────────────
    "data_root":      "data/processed/myodigitsv1",         # used when cv=standard
    "loso_data_root": "data/processed/myodigitsv1_loso_v4",   # 16ch, CORRECTED 60Hz notch
    "cv":             "loso",   # "loso" | "standard"
    "n_users":        21,
    "seq_len":        64,       # 3.35s context

    # ── proxy mode — fast 5-fold architecture screening ───────────────────────
    # Set proxy=True to run only 5 representative folds instead of all 21.
    # Folds chosen from run06 full LOSO ranking (21 users):
    #   User05 = best  (rank  1, MAE=3.77)
    #   User06 = med-  (rank  9, MAE=7.20)
    #   User13 = exact median (rank 11, MAE=7.36)
    #   User16 = med+  (rank 13, MAE=7.74)
    #   User12 = worst (rank 21, MAE=9.99)
    "proxy":          False,         # FULL 21-fold LOSO
    "proxy_folds":    [12, 5, 13],   # worst first, then best, then median
    "proxy_epochs":   60,
    "proxy_patience": 20,

    # ── resume (leave None for a fresh run) ───────────────────────────────────
    # To resume an interrupted run, set MD_RESUME_RUN to its directory name under
    # experiments/. Folds that already have metrics.json are skipped, and the
    # summary is rebuilt from every fold on disk, so start_fold only orders the
    # sweep; MD_START_FOLD=1 is always safe.
    "resume_run":     None,
    "start_fold":     1,
}


# ── environment overrides (used by orchestrator.sh for chained runs) ───────────
import os as _os
_OVERRIDES = {
    "MD_MODEL":       ("model",      str),
    "MD_TAG":         ("tag",        str),
    "MD_RESUME_RUN":  ("resume_run", lambda v: None if v in ("", "None", "none") else v),
    "MD_START_FOLD":  ("start_fold", int),
    "MD_IN_CH":       ("in_ch",      int),
    "MD_AUGMENT":     ("augment",    lambda v: v.lower() in ("1", "true", "yes", "on")),
    "MD_PRETRAINED":  ("pretrained_stem", lambda v: None if v in ("", "None", "none") else v),
    "MD_FREEZE_STEM": ("freeze_stem", lambda v: v.lower() in ("1", "true", "yes", "on")),
    "MD_N_TRAIN":     ("n_train_subjects", int),
    "MD_PROXY":       ("proxy",       lambda v: v.lower() in ("1", "true", "yes", "on")),
    "MD_PROXY_FOLDS": ("proxy_folds", lambda v: [int(x) for x in v.split(",") if x.strip()]),
    # locations; an absolute path is used as given, a relative one is taken from the repo root
    "MYODIGITS_DATA_ROOT":        ("loso_data_root",  str),
    "MYODIGITS_EXPERIMENTS_DIR":  ("experiments_dir", str),
}
for _env, (_key, _cast) in _OVERRIDES.items():
    _val = _os.environ.get(_env)
    if _val is not None:
        CONFIG[_key] = _cast(_val)
        print(f"[env override] {_key} = {CONFIG[_key]}")


MODEL_REGISTRY = {
    "cnn_v1":           ("myodigits.models.cnn_v1",           "CNN"),
    "cnn_v2":           ("myodigits.models.cnn_v2",           "CNN"),
    "lstm_v1":          ("myodigits.models.lstm_v1",          "LSTM"),
    "posenet_seq":      ("myodigits.models.posenet_seq",      "PoseNetSeq"),
    "tcn_v1":           ("myodigits.models.tcn_v1",           "PoseNetTCN"),
    "posenet_tx":       ("myodigits.models.posenet_tx",       "PoseNetTx"),
    "posenet_seq_v2":     ("myodigits.models.posenet_seq_v2",     "PoseNetSeqV2"),
    "posenet_tx_v2":      ("myodigits.models.posenet_tx_v2",      "PoseNetTxV2"),
    "posenet_tx_noBN":    ("myodigits.models.posenet_tx_noBN",    "PoseNetTxNoBN"),
    "posenet_tx_noResid": ("myodigits.models.posenet_tx_noResid", "PoseNetTxNoResid"),
    "emgnet_stft":        ("myodigits.models.emgnet_stft",        "EMGNetAngles"),
    "posenet_tx_stft":    ("myodigits.models.posenet_tx_stft",    "PoseNetTxStft"),
}


def _build_model(ModelClass, cfg):
    kwargs = {}
    sig = inspect.signature(ModelClass.__init__)
    if "in_ch" in sig.parameters and cfg.get("in_ch") is not None:
        kwargs["in_ch"] = cfg["in_ch"]
    # Pass through extra model kwargs (e.g. base, tx_heads, tx_layers, tx_dim_ff)
    for k, v in (cfg.get("model_kwargs") or {}).items():
        if k in sig.parameters:
            kwargs[k] = v
    return ModelClass(**kwargs)


def _maybe_load_pretrained(model, cfg):
    """Phase 3 transfer: load EgoEMG-pretrained stem+blocks into PoseNetTx.
    freeze_stem=True -> backbone weights frozen, only transformer+head train."""
    import torch as _t
    path = cfg.get("pretrained_stem")
    if not path:
        return model
    sd = _t.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"pretrained load: {len(unexpected)} UNEXPECTED keys "
                           f"(model/arch mismatch): {unexpected[:5]}")
    loaded = [k for k in sd if not k.startswith(tuple(m for m in missing))]
    print(f"[pretrained] loaded {len(sd)} stem/blocks tensors from {path}")
    if cfg.get("freeze_stem", False):
        nf = 0
        for name, p in model.named_parameters():
            if name.startswith(("stem.", "blocks.")):
                p.requires_grad_(False); nf += 1
        print(f"[pretrained] FROZE {nf} stem/blocks param tensors (train transformer+head only)")
    return model


def _next_run_number(experiments_dir):
    existing = [p.name for p in experiments_dir.iterdir()
                if p.is_dir() and p.name.startswith("run")]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[0][3:]))
        except (ValueError, IndexError):
            pass
    return max(nums, default=0) + 1


def _run_name(n, cfg):
    lr_str = f"{cfg['lr']:.0e}".replace("-0", "-")
    epochs = cfg.get("proxy_epochs", cfg["epochs"]) if cfg.get("proxy") else cfg["epochs"]
    suffix = "_loso" if cfg.get("cv") == "loso" else ""
    suffix += "_proxy" if cfg.get("proxy") else ""
    if cfg.get("tag"):
        suffix += f"_{cfg['tag']}"
    return f"run{n:02d}_{cfg['model']}_lr{lr_str}_bs{cfg['batch_size']}_ep{epochs}{suffix}"


def _make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, pin_memory=True, persistent_workers=True)


def main():
    if CONFIG.get("cv") == "loso":
        _main_loso()
    else:
        _main_standard()


# ── standard single-split training ───────────────────────────────────────────

def _main_standard():
    from myodigits.data.dataset import MyoDigitsDataset, SeqDataset
    from myodigits.training.train import train
    from myodigits.training.eval import run_eval

    model_name = CONFIG["model"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    module_path, class_name = MODEL_REGISTRY[model_name]
    ModelClass = getattr(importlib.import_module(module_path), class_name)

    experiments_dir = ROOT / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    run_n    = _next_run_number(experiments_dir)
    run_name = _run_name(run_n, CONFIG)
    exp_dir  = experiments_dir / run_name
    exp_dir.mkdir()
    print(f"Experiment: {run_name}")
    print(f"Folder:     {exp_dir}\n")

    config = {
        "epochs":         CONFIG["epochs"],
        "lr":             CONFIG["lr"],
        "batch_size":     CONFIG["batch_size"],
        "patience":       CONFIG["patience"],
        "weight_decay":   CONFIG["weight_decay"],
        "checkpoint_dir": str(exp_dir / "checkpoints"),
    }

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / CONFIG["data_root"]
    seq_len   = CONFIG.get("seq_len")

    if seq_len:
        ds_train = SeqDataset(data_root / "train", seq_len=seq_len)
        ds_val   = SeqDataset(data_root / "val",   seq_len=seq_len)
        ds_test  = SeqDataset(data_root / "test",  seq_len=seq_len)
    else:
        ds_train = MyoDigitsDataset(data_root / "train")
        ds_val   = MyoDigitsDataset(data_root / "val")
        ds_test  = MyoDigitsDataset(data_root / "test")

    loader_train = _make_loader(ds_train, config["batch_size"], shuffle=True)
    loader_val   = _make_loader(ds_val,   config["batch_size"], shuffle=False)
    loader_test  = _make_loader(ds_test,  config["batch_size"], shuffle=False)

    model = _build_model(ModelClass, CONFIG).to(device)
    print(f"Model: {ModelClass.__name__}  |  device: {device}\n")

    best_mae, history = train(model, loader_train, loader_val, config, device)
    run_eval(model, loader_test, history, config, device, exp_dir, run_name)


# ── leave-one-subject-out cross-validation ───────────────────────────────────

def _main_loso():
    from myodigits.data.dataset import (MyoDigitsDataset, LosoDataset, SeqDataset,
                                         LosoSeqDataset, StftDataset, LosoStftDataset)
    from myodigits.training.train import train
    from myodigits.training.eval import run_eval

    model_name = CONFIG["model"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    module_path, class_name = MODEL_REGISTRY[model_name]
    ModelClass = getattr(importlib.import_module(module_path), class_name)

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loso_data_root = ROOT / CONFIG["loso_data_root"]
    n_users        = CONFIG["n_users"]
    all_users      = list(range(1, n_users + 1))
    seq_len        = CONFIG.get("seq_len")

    # load per-user joint denorm stats if the dataset has normalized joints (v2)
    _loso_meta_path = loso_data_root / "meta.json"
    _loso_meta = json.loads(_loso_meta_path.read_text()) if _loso_meta_path.exists() else {}
    _per_user_stats = _loso_meta.get("per_user", {})

    experiments_dir = ROOT / CONFIG.get("experiments_dir", "experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)

    proxy         = CONFIG.get("proxy", False)
    proxy_folds   = list(CONFIG.get("proxy_folds", [5, 9, 12]))
    proxy_epochs  = CONFIG.get("proxy_epochs", 30)
    proxy_patience= CONFIG.get("proxy_patience", 12)

    # resume works in both full + proxy mode; the per-fold metrics.json skip-guard
    # below makes already-completed folds no-ops, so a resumed run continues where
    # it left off (start_fold only orders the full-LOSO sweep).
    resume_run  = CONFIG.get("resume_run")
    start_fold  = 1 if proxy else CONFIG.get("start_fold", 1)

    if resume_run:
        exp_dir  = experiments_dir / resume_run
        run_name = resume_run
        if not exp_dir.exists():
            raise FileNotFoundError(f"resume_run directory not found: {exp_dir}")
        print(f"Resuming: {run_name}  (starting from fold {start_fold})")
    else:
        run_n    = _next_run_number(experiments_dir)
        run_name = _run_name(run_n, CONFIG)
        exp_dir  = experiments_dir / run_name
        exp_dir.mkdir()
        start_fold = 1

    print(f"Folder:     {exp_dir}")
    if proxy:
        print(f"PROXY mode: folds {proxy_folds}  |  epochs={proxy_epochs}  |  patience={proxy_patience}")
    print(f"LOSO CV: {n_users} folds  |  model: {model_name}  |  device: {device}\n")

    all_fold_metrics = []

    # proxy: iterate in the exact order of proxy_folds (e.g. worst first)
    # full LOSO: iterate all users in order, respecting start_fold
    loop_users = list(proxy_folds) if proxy else [
        u for u in all_users if all_users.index(u) + 1 >= start_fold
    ]

    for test_user in loop_users:
        fold_idx = all_users.index(test_user)
        # 3 val users: next 3 in the circular list after the test user
        val_users   = [all_users[(fold_idx + 1 + k) % n_users] for k in range(3)]
        train_users = [u for u in all_users if u != test_user and u not in val_users]

        # Subject-count experiment: subsample training subjects to N (val/test unchanged).
        _ntr = CONFIG.get("n_train_subjects")
        if _ntr and _ntr < len(train_users):
            _rng = np.random.RandomState(1234 + fold_idx)   # deterministic per fold
            train_users = sorted(_rng.choice(train_users, size=_ntr, replace=False).tolist())
            print(f"  [subject-count] using {_ntr} training subjects: {train_users}")

        print(f"{'='*60}")
        print(f"Fold {fold_idx+1:02d}/{n_users}  |  "
              f"test=User{test_user:02d}  "
              f"val={[f'User{v:02d}' for v in val_users]}  "
              f"train={len(train_users)} users")
        print(f"{'='*60}")

        fold_dir = exp_dir / f"fold{fold_idx+1:02d}_testUser{test_user:02d}"
        fold_dir.mkdir(exist_ok=True)

        if (fold_dir / "metrics.json").exists():
            print(f"  [skip] metrics.json found — fold already complete.\n")
            continue

        train_dirs = [loso_data_root / f"User{u}" for u in train_users]
        val_dirs   = [loso_data_root / f"User{u}" for u in val_users]
        test_dir   = loso_data_root / f"User{test_user}"

        augment = CONFIG.get("augment", False)
        if CONFIG.get("stft", False):
            win_hop  = CONFIG.get("stft_win_hop", 256)
            ds_train = LosoStftDataset(train_dirs, win_hop=win_hop)
            ds_val   = LosoStftDataset(val_dirs,   win_hop=win_hop)
            ds_test  = StftDataset(test_dir, win_hop=win_hop, verbose=False)
        elif seq_len and STRICT_NORM:
            _sn_meta  = _sn_load_meta(loso_data_root)
            _sn_stats = _sn_fold_stats(_sn_meta, train_users,
                                       exclude_users=[test_user] + val_users)
            _sn_save(fold_dir / "strict_norm_stats.json", _sn_stats)
            ds_train = StrictLosoSeqDataset(train_dirs, _sn_meta, _sn_stats,
                                            seq_len=seq_len, augment=augment, tag="/train")
            ds_val   = StrictLosoSeqDataset(val_dirs, _sn_meta, _sn_stats,
                                            seq_len=seq_len, augment=False, tag="/val")
            ds_test  = StrictSeqDataset(test_dir, _sn_meta, _sn_stats, seq_len=seq_len)
        elif seq_len:
            ds_train = LosoSeqDataset(train_dirs, seq_len=seq_len, augment=augment)
            ds_val   = LosoSeqDataset(val_dirs,   seq_len=seq_len, augment=False)
            ds_test  = SeqDataset(test_dir, seq_len=seq_len, verbose=False)
        else:
            ds_train = LosoDataset(train_dirs)
            ds_val   = LosoDataset(val_dirs)
            ds_test  = MyoDigitsDataset(test_dir, verbose=False)

        print(f"  train: {len(ds_train):,}  val: {len(ds_val):,}  test: {len(ds_test):,} windows\n")

        fold_config = {
            "epochs":         proxy_epochs   if proxy else CONFIG["epochs"],
            "patience":       proxy_patience if proxy else CONFIG["patience"],
            "lr":             CONFIG["lr"],
            "batch_size":     CONFIG["batch_size"],
            "weight_decay":   CONFIG["weight_decay"],
            "smooth_lambda":  CONFIG.get("smooth_lambda", 0.0),
            "checkpoint_dir": str(fold_dir / "checkpoints"),
        }

        loader_train = _make_loader(ds_train, fold_config["batch_size"], shuffle=True)
        loader_val   = _make_loader(ds_val,   fold_config["batch_size"], shuffle=False)
        loader_test  = _make_loader(ds_test,  fold_config["batch_size"], shuffle=False)

        model = _build_model(ModelClass, CONFIG).to(device)
        model = _maybe_load_pretrained(model, CONFIG).to(device)

        best_mae, history = train(model, loader_train, loader_val, fold_config, device)

        fold_label   = f"{run_name}_fold{fold_idx+1:02d}_testUser{test_user:02d}"
        if STRICT_NORM:
            # degrees come from the TRAINING participants statistics, never the held-out one
            user_stats = {"jnt_mean": _sn_stats["jnt_mean"], "jnt_std": _sn_stats["jnt_std"]}
        else:
            user_stats = _per_user_stats.get(f"User{test_user}", {})
        jnt_denorm   = (
            np.array(user_stats["jnt_mean"], dtype=np.float32),
            np.array(user_stats["jnt_std"],  dtype=np.float32),
        ) if "jnt_mean" in user_stats else None
        fold_metrics = run_eval(model, loader_test, history, fold_config, device,
                                fold_dir, fold_label, jnt_denorm=jnt_denorm)
        fold_metrics["fold"]       = fold_idx + 1
        fold_metrics["test_user"]  = test_user
        fold_metrics["val_users"]  = val_users
        all_fold_metrics.append(fold_metrics)

        print(f"\nFold {fold_idx+1:02d} done.  test_mae={fold_metrics['test_mae']:.4f}\n")

        # Free persistent workers and dataset tensors before next fold loads its data
        del loader_train, loader_val, loader_test, ds_train, ds_val, ds_test
        gc.collect()

    # The summary is rebuilt from every per-fold metrics.json on disk, never
    # from this process's in-memory list and never from an earlier summary.
    # Up to commit 8dd416b a run killed mid-loop wrote no summary, and the old
    # merge (which required one to exist) silently dropped earlier folds on the
    # next resume. See src/myodigits/analysis/loso_summary.py.
    from myodigits.analysis.loso_summary import expected_folds_for, save_loso_summary
    expected_folds = expected_folds_for(n_users, proxy=proxy, proxy_folds=proxy_folds,
                                        all_users=all_users)
    save_loso_summary(exp_dir, run_name, expected_folds=expected_folds)


if __name__ == "__main__":
    main()
