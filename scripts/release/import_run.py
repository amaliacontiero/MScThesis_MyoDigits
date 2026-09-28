#!/usr/bin/env python3
"""Import a completed LOSO run directory into ``results/`` for publication.

Copies only small, non-identifying artifacts:

  * ``foldNN_testUserNN_metrics.json`` for every fold (byte-for-byte);
  * ``strict_norm_stats/foldNN_testUserNN_strict_norm_stats.json`` if present;
  * a sanitised ``PROVENANCE.json`` (see ``sanitise``);
  * ``loso_summary.json`` rebuilt from the copied fold files with the canonical
    ``myodigits.analysis.loso_summary`` code. Any summary inside the run
    directory is ignored, because historical summaries can be wrong.

Checkpoints, predictions, plots, notebooks and videos are never copied.

The import refuses a run that does not have every expected fold, so a partial
run cannot be published by accident.

Usage
    python scripts/release/import_run.py RUN_DIR --name strict_norm_tx --n-folds 21
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from myodigits.analysis.loso_summary import (  # noqa: E402
    build_summary, collect_fold_metrics, materialise_release_layout)

# Paths inside the private working tree -> paths in this repository.
SOURCE_PATH_MAP = {
    "launch.py": "launch.py",
    "scripts/strict_norm.py": "scripts/normalisation/strict_norm.py",
    "scripts/strict_datasets.py": "scripts/normalisation/strict_datasets.py",
    "scripts/rebuild_loso_summary.py": "scripts/analysis/rebuild_loso_summary.py",
    "src/myodigits/training/eval.py": "src/myodigits/training/eval.py",
    "src/myodigits/training/train.py": "src/myodigits/training/train.py",
    "src/myodigits/data/dataset.py": "src/myodigits/data/dataset.py",
}

KEPT_KEYS = ("purpose", "start_time_utc", "launch_command", "environment_variables",
             "resolved_config", "fold_role_logic", "normalisation_protocol",
             "data_meta_sha256")


def sanitise(prov: dict) -> dict:
    """Keep what identifies the experiment; drop what identifies the machine.

    Dropped: hostname, processor string, full platform string, the private
    working repository's commit, remote URL and dirty-file listing, and the full
    ``pip freeze`` (which names the private repository). Source files are
    identified by SHA-256, which is what reproducibility actually needs.
    """
    out = {k: prov[k] for k in KEPT_KEYS if k in prov}
    out["source_hashes"] = [
        {"path_in_this_repository": SOURCE_PATH_MAP.get(h["path"], h["path"]),
         "path_at_run_time": h["path"], "sha256": h["sha256"], "bytes": h.get("bytes")}
        for h in prov.get("source_hashes", [])
    ]
    t = prov.get("torch", {})
    pkgs = {}
    for line in str(prov.get("package_environment", "")).splitlines():
        name = line.split("==")[0].strip().lower()
        if name in ("numpy", "scipy", "torch") and "==" in line:
            pkgs[name] = line.split("==", 1)[1].strip()
    out["environment"] = {
        "python": str(prov.get("platform", {}).get("python", "")).split()[0],
        "torch": t.get("torch"), "cuda": t.get("cuda_version"), "cudnn": t.get("cudnn"),
        "gpu": t.get("device_name"),
        "os": "Linux (WSL2)" if "WSL2" in str(prov.get("platform", {}).get("platform", "")) else None,
        "packages": pkgs,
    }
    out["sanitised"] = ("Machine identifiers, private repository metadata and the full "
                        "package listing were removed on import. Nothing affecting the "
                        "experiment was changed.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--name", required=True, help="results/<name>")
    ap.add_argument("--n-folds", type=int, default=21)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    run_name = run_dir.name
    records, problems = collect_fold_metrics(run_dir)
    found = sorted(r["fold"] for r in records)
    if problems or found != list(range(1, args.n_folds + 1)):
        print(f"REFUSED: folds found {found}, problems {problems}")
        return 1

    dest = REPO / "results" / args.name
    if dest.exists():
        print(f"REFUSED: {dest.relative_to(REPO)} already exists")
        return 1
    (dest / "strict_norm_stats").mkdir(parents=True)

    for fold_dir in sorted(run_dir.glob("fold*_testUser*")):
        shutil.copyfile(fold_dir / "metrics.json", dest / f"{fold_dir.name}_metrics.json")
        sn = fold_dir / "strict_norm_stats.json"
        if sn.exists():
            shutil.copyfile(sn, dest / "strict_norm_stats" / f"{fold_dir.name}_strict_norm_stats.json")
    if not any((dest / "strict_norm_stats").iterdir()):
        (dest / "strict_norm_stats").rmdir()

    prov_path = run_dir / "PROVENANCE.json"
    if prov_path.exists():
        prov = sanitise(json.loads(prov_path.read_text()))
        (dest / "PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        materialise_release_layout(dest, tmp)
        summary = build_summary(tmp, run_name, expected_folds=args.n_folds)
    (dest / "loso_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"imported {run_name} -> results/{args.name}: {summary['n_folds']} folds, "
          f"complete={summary['aggregation']['complete']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
