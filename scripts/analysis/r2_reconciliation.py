#!/usr/bin/env python3
"""Reproduce thesis Appendix D.3: three R² variants for the five configurations.

Runs from a clone. Reads the published per-joint sufficient statistics in
``results/analysis/r2/<config>.json`` (produced from private predictions by
``extract_r2_statistics.py``) and writes ``results/analysis/r2_reconciliation.json``.

Every value comes from ``myodigits.analysis.r2_variants``, the single
implementation used everywhere in this repository. Each variant is computed
within a held-out participant and averaged uniformly over the 21 folds.

Usage
    python scripts/analysis/r2_reconciliation.py           # write
    python scripts/analysis/r2_reconciliation.py --check   # fail if the file is stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from myodigits.analysis import r2_variants as r2  # noqa: E402

STATS = REPO / "results" / "analysis" / "r2"
OUT = REPO / "results" / "analysis" / "r2_reconciliation.json"

CONFIGS = {
    "myodigits_tx": "MyoDigits-TX",
    "myodigits_gru": "MyoDigits-GRU",
    "myodigits_tcn": "MyoDigits-TCN",
    "transfer_egoemg_finetuned": "Pretrained, fine-tuned",
    "transfer_egoemg_frozen": "Pretrained, parameters frozen",
}


def build() -> dict:
    out = {
        "analysis": "coefficient-of-determination variants",
        "thesis_table": "Appendix D, Table D.2 (Section D.3)",
        "definitions": {
            r2.JOINT_CENTRED: "1 - sum_j SSE_j / sum_j SST_j, each joint about its own mean",
            r2.UNIFORM_MEAN: "(1/J) sum_j [1 - SSE_j / SST_j]",
            r2.GLOBALLY_POOLED: "1 - sum_j SSE_j / SST about one grand mean over all joints and samples",
        },
        "labels": r2.LABELS,
        "aggregation": "within each held-out participant, then the uniform mean over 21 folds",
        "reported_in_thesis": r2.JOINT_CENTRED,
        "configurations": {},
    }
    means = {}
    for key, label in CONFIGS.items():
        src = json.loads((STATS / f"{key}.json").read_text())
        folds = sorted(src["folds"], key=lambda f: f["fold"])
        if [f["fold"] for f in folds] != list(range(1, 22)):
            raise SystemExit(f"{key}: expected folds 1-21")
        per_fold = []
        between_share = []
        for f in folds:
            v = r2.r2_from_statistics(f)
            per_fold.append({"fold": f["fold"], **v})
            sst_j = float(np.sum(f["sst_per_joint"]))
            between_share.append(1.0 - sst_j / r2.global_total_sum_of_squares(f))
        maes = [json.loads((REPO / "results" / key / f"fold{k:02d}_testUser{k:02d}_metrics.json")
                           .read_text())["test_mae"] for k in range(1, 22)]
        agg = {}
        for var in r2.VARIANTS:
            vals = np.array([p[var] for p in per_fold])
            agg[var] = {"loso_mean": r2.loso_mean(per_fold, var),
                        "sd_ddof0": float(vals.std(ddof=0)),
                        "min": float(vals.min()), "max": float(vals.max())}
        means[key] = {var: agg[var]["loso_mean"] for var in r2.VARIANTS}
        out["configurations"][key] = {
            "label": label,
            "mae_deg_loso_mean": float(np.mean(maes)),
            "variants": agg,
            "between_joint_share_of_pooled_sst_mean": float(np.mean(between_share)),
            "per_fold": per_fold,
        }

    ordering = {var: sorted(CONFIGS, key=lambda k: -means[k][var]) for var in r2.VARIANTS}
    mae = {k: out["configurations"][k]["mae_deg_loso_mean"] for k in CONFIGS}
    ordering["mae_best_first"] = sorted(CONFIGS, key=lambda k: mae[k])
    ft, fz = "transfer_egoemg_finetuned", "transfer_egoemg_frozen"
    out["ordering_best_first"] = ordering
    out["transfer_condition_reversal"] = {
        "note": "both transfer conditions share an identical architecture, so this is a "
                "transfer-condition ordering, not an architecture ordering",
        "finetuned_above_frozen_under": {
            var: bool(means[ft][var] > means[fz][var]) for var in r2.VARIANTS},
        "finetuned_better_by_mae": bool(mae[ft] < mae[fz]),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed file")
    args = ap.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != text:
            print(f"STALE: {OUT.relative_to(REPO)} differs from a fresh computation")
            return 1
        print(f"up to date: {OUT.relative_to(REPO)}")
        return 0
    OUT.write_text(text)
    r = json.loads(text)
    print(f"{'configuration':32s} {'MAE':>6s} {'jc agg':>8s} {'uniform':>8s} {'pooled':>8s}")
    for k, c in r["configurations"].items():
        v = c["variants"]
        print(f"{c['label']:32s} {c['mae_deg_loso_mean']:6.2f} "
              f"{v[r2.JOINT_CENTRED]['loso_mean']:8.4f} {v[r2.UNIFORM_MEAN]['loso_mean']:8.4f} "
              f"{v[r2.GLOBALLY_POOLED]['loso_mean']:8.4f}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
