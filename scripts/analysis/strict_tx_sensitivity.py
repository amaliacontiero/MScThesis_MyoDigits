#!/usr/bin/env python3
"""Strict-normalisation sensitivity comparison for MyoDigits-TX (thesis Section 11.9).

Runs from a clone. Reads only published per-fold records:

    results/myodigits_tx/       participant-calibrated protocol (primary result)
    results/strict_norm_tx/     training-participant statistics only

and writes ``results/analysis/strict_tx_sensitivity.json``.

WHAT THIS COMPARISON IS
    Two separately trained runs of the same architecture, participant splits,
    training procedure and hyperparameters. The intended design difference is
    the standardisation protocol. No global training seed was fixed and neither
    configuration was repeated, so the difference also contains uncontrolled
    variation from weight initialisation, batch ordering and early stopping.

    It is a separately trained sensitivity comparison. It is not a controlled
    ablation, not an isolated causal effect of the protocol, and not an exact
    calibration penalty. The paired interval and Wilcoxon test quantify
    consistency across held-out participants for these two trained runs; they
    do not include uncertainty across repeated model training.

It also re-verifies, from the published fold statistics, that every strict fold
fitted its statistics on exactly the 17 training participants given by the fold
rule, with the test and all three validation participants excluded.

Usage
    python scripts/analysis/strict_tx_sensitivity.py           # write
    python scripts/analysis/strict_tx_sensitivity.py --check   # fail if the file is stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from myodigits.analysis import paired  # noqa: E402

CALIBRATED = REPO / "results" / "myodigits_tx"
STRICT = REPO / "results" / "strict_norm_tx"
OUT = REPO / "results" / "analysis" / "strict_tx_sensitivity.json"
N_USERS = 21


def fold_roles(test_user: int, n_users: int = N_USERS):
    """Identical to launch.py: validation = the next three participants, circularly."""
    all_users = list(range(1, n_users + 1))
    i = all_users.index(test_user)
    val = [all_users[(i + 1 + k) % n_users] for k in range(3)]
    train = [u for u in all_users if u != test_user and u not in val]
    return val, train


def per_participant_mae(results_dir: Path) -> dict:
    out = {}
    for f in sorted(results_dir.glob("fold*_testUser*_metrics.json")):
        user = int(f.name.split("testUser")[1][:2])
        out[user] = float(json.loads(f.read_text())["test_mae"])
    return out


def verify_strict_fold_roles() -> dict:
    rows, failures = [], []
    for u in range(1, N_USERS + 1):
        p = STRICT / "strict_norm_stats" / f"fold{u:02d}_testUser{u:02d}_strict_norm_stats.json"
        s = json.loads(p.read_text())
        val, train = fold_roles(u)
        fitted = sorted(int(x) for x in s["train_users"])
        excluded = sorted(int(x) for x in s["excluded_users"])
        ok = (fitted == train and excluded == sorted([u] + val)
              and u not in fitted and not set(val) & set(fitted)
              and s["n_train_participants"] == 17
              and s["weighting"] == "participant-weighted")
        rows.append({"fold": u, "test_user": u, "validation_users": val,
                     "n_fitting_participants": len(fitted), "roles_verified": ok})
        if not ok:
            failures.append(u)
    return {"all_folds_verified": not failures, "failed_folds": failures, "folds": rows}


def build() -> dict:
    cal, strict = per_participant_mae(CALIBRATED), per_participant_mae(STRICT)
    users = sorted(set(cal) & set(strict))
    if users != list(range(1, N_USERS + 1)):
        raise SystemExit(f"expected paired participants 1-{N_USERS}, got {users}")

    a = np.array([cal[u] for u in users])
    b = np.array([strict[u] for u in users])
    d = b - a

    return {
        "analysis": "strict-normalisation sensitivity comparison, MyoDigits-TX",
        "thesis_section": "11.9",
        "design": {
            "calibrated_results": "results/myodigits_tx",
            "strict_results": "results/strict_norm_tx",
            "type": "separately trained sensitivity comparison",
            "not": ["a controlled ablation", "an isolated causal effect of the protocol",
                    "an exact calibration penalty"],
            "intended_difference": "standardisation protocol",
            "uncontrolled_variation": ["weight initialisation", "batch ordering",
                                       "early-stopping epoch"],
            "training_seeds": "not fixed and not repeated in either run",
            "inference_scope": "the paired interval and Wilcoxon test quantify consistency "
                               "across held-out participants for these two trained runs; they "
                               "do not include uncertainty across repeated model training",
            "calibrated_protocol": "each participant standardised with that participant's own "
                                   "statistics; predictions returned to degrees with the "
                                   "held-out participant's own reference statistics",
            "strict_protocol": "input and target statistics fitted on the 17 training "
                               "participants of each fold, participant-weighted, applied "
                               "unchanged to validation and test participants; predictions "
                               "returned to degrees with those training statistics",
            "summary_input": "the 21-fold summary rebuilt from per-fold records; the "
                             "13-fold summary written by the interrupted run is not used",
        },
        "per_participant": [
            {"participant": u, "mae_calibrated_deg": float(x), "mae_strict_deg": float(y),
             "paired_difference_deg": float(y - x)}
            for u, x, y in zip(users, a, b)
        ],
        "calibrated": paired.describe(a),
        "strict": paired.describe(b),
        "paired": {
            "mean_difference_deg": float(d.mean()),
            "bootstrap_ci": paired.bootstrap_mean_ci(d),
            "wilcoxon": paired.paired_wilcoxon(b, a),
            "participants_higher_under_strict": int((d > 0).sum()),
            "participants_total": int(len(d)),
            "min_difference_deg": float(d.min()),
            "max_difference_deg": float(d.max()),
        },
        "association_calibrated_vs_strict": paired.associations(a, b),
        "strict_fold_role_verification": verify_strict_fold_roles(),
        "statements": [
            f"Under the separately trained strict protocol, mean MAE was {b.mean():.2f} degrees "
            f"against {a.mean():.2f} degrees under the participant-calibrated protocol.",
            f"MAE was higher under the strict protocol for all {int((d > 0).sum())} of "
            f"{len(d)} held-out participants.",
            "Per-participant error remained strongly associated across the two protocols; "
            "the rankings were similar rather than identical.",
        ],
    }


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
    p = r["paired"]
    print(f"calibrated {r['calibrated']['mean']:.4f}  strict {r['strict']['mean']:.4f}  "
          f"diff {p['mean_difference_deg']:.4f}  CI [{p['bootstrap_ci']['low']:.3f}, "
          f"{p['bootstrap_ci']['high']:.3f}]  W={p['wilcoxon']['W']:.0f} "
          f"p={p['wilcoxon']['p']:.3e}  higher {p['participants_higher_under_strict']}/21")
    s = r["association_calibrated_vs_strict"]
    print(f"Pearson r {s['pearson_r']:.4f}  Spearman rho {s['spearman_rho']:.4f}  "
          f"fold roles verified: {r['strict_fold_role_verification']['all_folds_verified']}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
