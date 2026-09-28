"""CPU smoke tests that need neither a GPU nor the private dataset.

A synthetic dataset in the real on-disk format (``UserN/seg_*.npz`` holding
``X`` (windows, 16, 400) and ``y`` (windows, 20), plus ``meta.json``) is written
to a temporary directory, and the real ``launch.py`` LOSO path trains and
evaluates tiny proxy runs on it in a subprocess.

Two test-only patches keep this fast: data loaders use no worker processes, and
the hand-animation video is not rendered (it needs ffmpeg and is not a metric).
Nothing about training, splitting, evaluation, checkpointing or early stopping
is patched.
"""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]

# Thesis Table 4.2, trainable parameters.
THESIS_PARAMETERS = {"posenet_tx": 534_868, "posenet_seq": 838_804, "tcn_v1": 348_308}


@pytest.fixture
def launch(monkeypatch):
    for k in list(os.environ):
        if k.startswith("MD_") or k.startswith("MYODIGITS_"):
            monkeypatch.delenv(k, raising=False)
    sys.modules.pop("launch", None)
    monkeypatch.syspath_prepend(str(REPO))
    return importlib.import_module("launch")


def test_launch_defaults_start_a_fresh_full_run(launch):
    c = launch.CONFIG
    assert c["resume_run"] is None and c["start_fold"] == 1
    assert c["proxy"] is False and c["n_users"] == 21 and c["cv"] == "loso"
    assert (c["epochs"], c["lr"], c["batch_size"], c["patience"], c["weight_decay"]) == \
        (60, 3e-4, 64, 20, 1e-2)
    assert c["seq_len"] == 64 and c["in_ch"] == 16 and c["augment"] is True


@pytest.mark.parametrize("model_name", sorted(THESIS_PARAMETERS))
def test_models_match_thesis_parameter_counts_and_shapes(launch, model_name):
    module_path, class_name = launch.MODEL_REGISTRY[model_name]
    cls = getattr(importlib.import_module(module_path), class_name)
    cfg = dict(launch.CONFIG, model=model_name)
    model = launch._build_model(cls, cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == THESIS_PARAMETERS[model_name]
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 4, 16, 400))
    assert tuple(out.shape) == (2, 4, 20)


def test_fold_rule_and_strict_statistics_exclude_held_out_participants():
    from strict_norm import fold_statistics
    all_users = list(range(1, 22))
    meta = {"per_user": {f"User{u}": {"emg_mean": [0.0] * 16, "emg_std": [1.0] * 16,
                                      "jnt_mean": [float(u)] * 20, "jnt_std": [2.0] * 20,
                                      "windows": 100} for u in all_users}}
    for test_user in all_users:
        i = all_users.index(test_user)
        val = [all_users[(i + 1 + k) % 21] for k in range(3)]
        train = [u for u in all_users if u != test_user and u not in val]
        s = fold_statistics(meta, train, exclude_users=[test_user] + val)
        assert s["n_train_participants"] == 17
        assert not ({test_user, *val} & set(s["train_users"]))
    with pytest.raises(AssertionError, match="LEAK"):
        fold_statistics(meta, [1, 2, 3], exclude_users=[3])


RUNNER = r'''
import os, sys, json
sys.path.insert(0, {repo!r})
import launch
launch.CONFIG.update(n_users=6, seq_len=4, batch_size=4, epochs=1, patience=1,
                     proxy_epochs=1, proxy_patience=1)
launch._make_loader = lambda ds, bs, shuffle: __import__("torch").utils.data.DataLoader(
    ds, batch_size=bs, shuffle=shuffle, num_workers=0)
import myodigits.training.eval as ev
ev._save_hand_animation = lambda **kw: None
launch.main()
'''


def _write_synthetic_dataset(root: Path, n_users=6, segs=2, windows=12):
    rng = np.random.default_rng(0)
    meta = {"per_user": {}}
    for u in range(1, n_users + 1):
        d = root / f"User{u}"
        d.mkdir(parents=True)
        for s in range(segs):
            X = rng.standard_normal((windows, 16, 400)).astype(np.float32)
            y = rng.standard_normal((windows, 20)).astype(np.float32)
            np.savez(d / f"seg_{s:03d}.npz", X=X, y=y)
        meta["per_user"][f"User{u}"] = {
            "emg_mean": rng.normal(0, 0.01, 16).tolist(), "emg_std": rng.uniform(0.02, 0.05, 16).tolist(),
            "jnt_mean": rng.uniform(0, 50, 20).tolist(), "jnt_std": rng.uniform(3, 25, 20).tolist(),
            "windows": segs * windows}
    (root / "meta.json").write_text(json.dumps(meta))


def _run(tmp_path, extra_env):
    script = tmp_path / "runner.py"
    script.write_text(RUNNER.format(repo=str(REPO)))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MD_", "MYODIGITS_"))}
    env.update(CUDA_VISIBLE_DEVICES="", MYODIGITS_DATA_ROOT=str(tmp_path / "data"),
               MYODIGITS_EXPERIMENTS_DIR=str(tmp_path / "experiments"), MPLBACKEND="Agg")
    env.update(extra_env)
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_end_to_end_proxy_run_then_safe_resume(tmp_path):
    _write_synthetic_dataset(tmp_path / "data")
    out = _run(tmp_path, {"MD_MODEL": "posenet_tx", "MD_TAG": "smoke", "MD_PROXY": "1",
                          "MD_PROXY_FOLDS": "3,1,2", "MD_RESUME_RUN": "none"})
    assert "device: cpu" in out
    (run_dir,) = list((tmp_path / "experiments").iterdir())
    s = json.loads((run_dir / "loso_summary.json").read_text())
    assert s["n_folds"] == 3 and s["aggregation"]["complete"] is True
    assert s["aggregation"]["folds_found"] == [1, 2, 3]
    for k in (1, 2, 3):
        fd = run_dir / f"fold{k:02d}_testUser{k:02d}"
        assert (fd / "checkpoints" / "best.pt").exists() and (fd / "checkpoints" / "last.pt").exists()
        m = json.loads((fd / "metrics.json").read_text())
        assert m["n_test_windows"] == 2 * (12 - 4 + 1) * 4

    # Interrupt: fold 3 unfinished and no summary on disk, which is what defeated
    # the old merge. Resuming must retrain only fold 3 and still summarise all three.
    kept = {k: (run_dir / f"fold{k:02d}_testUser{k:02d}" / "metrics.json").read_bytes() for k in (1, 2)}
    (run_dir / "fold03_testUser03" / "metrics.json").unlink()
    (run_dir / "loso_summary.json").unlink()
    out2 = _run(tmp_path, {"MD_MODEL": "posenet_tx", "MD_TAG": "smoke", "MD_PROXY": "1",
                           "MD_PROXY_FOLDS": "3,1,2", "MD_RESUME_RUN": run_dir.name})
    assert out2.count("[skip] metrics.json found") == 2
    s2 = json.loads((run_dir / "loso_summary.json").read_text())
    assert s2["n_folds"] == 3 and s2["aggregation"]["complete"] is True
    for k, b in kept.items():
        assert (run_dir / f"fold{k:02d}_testUser{k:02d}" / "metrics.json").read_bytes() == b


def test_end_to_end_strict_normalisation_fold(tmp_path):
    _write_synthetic_dataset(tmp_path / "data")
    out = _run(tmp_path, {"MYODIGITS_STRICT_NORM": "1", "MD_MODEL": "posenet_seq",
                          "MD_TAG": "smokestrict", "MD_PROXY": "1", "MD_PROXY_FOLDS": "2",
                          "MD_RESUME_RUN": "none"})
    assert "[STRICT NORM]" in out
    (run_dir,) = list((tmp_path / "experiments").iterdir())
    st = json.loads((run_dir / "fold02_testUser02" / "strict_norm_stats.json").read_text())
    assert st["train_users"] == [1, 6] and st["excluded_users"] == [2, 3, 4, 5]
    s = json.loads((run_dir / "loso_summary.json").read_text())
    assert s["n_folds"] == 1 and s["aggregation"]["complete"] is True
