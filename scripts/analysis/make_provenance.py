#!/usr/bin/env python3
"""Immutable provenance snapshot for a strict-normalisation run.

Run with the SAME environment variables as the launch, so the resolved CONFIG
recorded here is exactly the one the run will use.

    python make_provenance.py OUTPUT.json "the exact launch command"
"""
# --- configurable roots -------------------------------------------------
# The analysis and figure scripts read run directories that are not
# distributed with this repository. Point these at your own copies.
import os as _os
from pathlib import Path as _Path
_ROOT   = _Path(_os.environ.get("MYODIGITS_ROOT", _Path(__file__).resolve().parents[2]))
_THESIS = _Path(_os.environ.get("MYODIGITS_THESIS_DIR", _ROOT))
_EGOEMG = _Path(_os.environ.get("MYODIGITS_EGOEMG_DIR", _ROOT.parent / "MD_EgoEMG"))
# ------------------------------------------------------------------------
import hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

ROOT = Path(str(_ROOT))
OUT = Path(sys.argv[1])
CMD = sys.argv[2] if len(sys.argv) > 2 else ""

TRACKED = [
    "launch.py",
    "scripts/normalisation/strict_norm.py",
    "scripts/normalisation/strict_datasets.py",
    "scripts/analysis/rebuild_loso_summary.py",
    "src/myodigits/training/eval.py",
    "src/myodigits/training/train.py",
    "src/myodigits/data/dataset.py",
]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<failed: {e}>"


def sha256(p):
    f = ROOT / p
    if not f.exists():
        return {"path": p, "sha256": None, "note": "MISSING"}
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    return {"path": p, "sha256": h, "bytes": f.stat().st_size,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(f.stat().st_mtime))}


# resolved configuration, under the current environment
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import launch  # noqa: E402

cfg = {k: v for k, v in launch.CONFIG.items()}

# the fold-role logic, verbatim from the source that will execute
src = (ROOT / "launch.py").read_text().splitlines()
fold_logic = []
for i, line in enumerate(src):
    if "val_users   =" in line or "train_users =" in line:
        fold_logic.append(f"L{i+1}: {line.strip()}")

git = {
    "is_repo": (ROOT / ".git").exists(),
    "commit": sh("git rev-parse HEAD"),
    "branch": sh("git rev-parse --abbrev-ref HEAD"),
    "dirty_files": sh("git status --porcelain"),
    "describe": sh("git describe --always --dirty"),
}

try:
    import torch
    torch_info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }
except Exception as e:
    torch_info = {"error": str(e)}

snapshot = {
    "purpose": "Strict-normalisation (Experiment A) 21-fold LOSO, MyoDigits-TX only.",
    "immutable": "Written once at launch. Do not edit.",
    "start_time_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "launch_command": CMD,
    "environment_variables": {k: v for k, v in sorted(os.environ.items())
                              if k.startswith(("MD_", "MYODIGITS_", "CUDA_", "PYTORCH_"))},
    "resolved_config": cfg,
    "fold_role_logic": fold_logic,
    "normalisation_protocol": {
        "statistics_fitted_on": "the 17 training participants of each fold only",
        "pooling": "participant-weighted (equal weight per participant)",
        "applied_to": "training, validation and test participants alike",
        "denormalisation_to_degrees": "training-participant statistics; the held-out "
                                      "participant's reference angles are never consulted",
        "exclusion_assertion": "fold_statistics() asserts test and validation participants "
                               "are absent from the fitting set",
    },
    "source_hashes": [sha256(p) for p in TRACKED],
    "git": git,
    "platform": {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    },
    "torch": torch_info,
    "package_environment": sh(f"{sys.executable} -m pip freeze"),
    "data_root": str(launch.CONFIG.get("loso_data_root")),
    "data_meta_sha256": hashlib.sha256(
        (ROOT / "data/processed/myodigitsv1_loso_v4/meta.json").read_bytes()).hexdigest(),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(snapshot, indent=2))
print(f"provenance snapshot -> {OUT}")
print(f"  start        : {snapshot['start_time_local']}")
print(f"  git commit   : {git['commit'][:12] if git['commit'] else 'n/a'}  "
      f"dirty={'yes' if git['dirty_files'] else 'no'}")
print(f"  strict_norm  : {[h['sha256'][:12] for h in snapshot['source_hashes'] if 'strict_norm' in h['path']]}")
print(f"  launch.py    : {[h['sha256'][:12] for h in snapshot['source_hashes'] if h['path'] == 'launch.py']}")
print(f"  model/config : {cfg['model']} {cfg['model_kwargs']} in_ch={cfg['in_ch']} seq_len={cfg['seq_len']}")
print(f"  proxy={cfg.get('proxy')} start_fold={cfg.get('start_fold')} resume_run={cfg.get('resume_run')}")
print(f"  fold logic   : {len(fold_logic)} lines captured")
print(f"  pip packages : {len(snapshot['package_environment'].splitlines())}")
