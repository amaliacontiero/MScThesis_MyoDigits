#!/usr/bin/env python3
"""Protocol-matched constant-posture baselines (thesis Table 7.4 and Table A.4).

WHAT THIS IS
    A constant twenty-joint posture, predicted for every scored position, is the
    floor a model has to beat. Two are reported, one per protocol, so that each
    model is compared with a baseline that had access to the same information.

    Strict LOSO baseline. The posture is the participant-balanced mean of the
    seventeen TRAINING participants of that fold: the mean over those
    participants of each participant's per-joint mean angle. The held-out
    participant and all three validation participants contribute nothing. This
    is the same pooling ``scripts/normalisation/strict_norm.py`` fits, so for
    every fold the posture equals the ``jnt_mean`` recorded in that fold's
    ``strict_norm_stats.json``, and that equality is asserted here.

    Participant-calibrated baseline. The posture is the held-out participant's
    OWN per-joint mean angle, the statistic that protocol already uses to return
    predictions to degrees. It is fitted on the held-out participant's reference
    labels, so it is TEST-LABEL-INFORMED: a diagnostic bound, not a label-free
    prediction and not a deployable baseline.

SCORING
    Both baselines are scored on exactly the positions the neural runs are
    scored on, using the sequence-position multiplicities of
    ``myodigits.analysis.weighting``. Scoring a constant on unique windows
    instead would evaluate it on a different set from the models.

WHICH MEAN
    ``meta.json``'s per-participant ``jnt_mean`` is the mean over every 2 kHz
    sample of the continuous angle series (``preprocess_loso_v4.py``), whereas a
    scored target is the last sample of its 400-sample window. The two differ by
    up to about 0.35 degrees per joint. The thesis uses the documented
    ``meta.json`` statistic for both baselines, so that is what is reported.
    ``--empirical`` additionally records the multiplicity-weighted mean of the
    windowed targets, which is the posture that minimises squared error on the
    scored set; it is reported only as a sensitivity figure and the strict
    baseline is unchanged in substance by the choice.

DATA
    Writing the record needs the processed dataset (``MYODIGITS_DATA_ROOT``,
    default ``data/processed/myodigitsv1_loso_v4``), which is not distributed.
    ``--verify`` needs no data: it recomputes every aggregate and every model
    reduction from the committed record and the published per-fold metrics, and
    is what the test suite and a data-free clone run.

Usage
    python scripts/analysis/posture_baselines.py            # write the record (needs data)
    python scripts/analysis/posture_baselines.py --check    # fail if stale (needs data)
    python scripts/analysis/posture_baselines.py --verify   # data-free consistency check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from myodigits.analysis.weighting import (  # noqa: E402
    SEQ_LEN, check_trial, window_multiplicity)

RESULTS = REPO / "results"
OUT = RESULTS / "analysis" / "posture_baselines.json"
DATA_ROOT = Path(os.environ.get("MYODIGITS_DATA_ROOT",
                                REPO / "data" / "processed" / "myodigitsv1_loso_v4"))
N_PARTICIPANTS = 21
J = 20

# Which published run stands for which protocol in the reduction table.
MODELS = {
    "training_statistics_only": [("strict_norm_tx", "MyoDigits-TX"),
                                 ("strict_norm_gru", "MyoDigits-GRU")],
    "participant_calibrated": [("myodigits_tx", "MyoDigits-TX"),
                               ("myodigits_gru", "MyoDigits-GRU")],
}
BASELINE_OF = {"training_statistics_only": "strict_posture_mae_deg",
               "participant_calibrated": "calibrated_posture_mae_deg"}


def fold_roles(test_user, n=N_PARTICIPANTS):
    """launch.py's rule: the next three participants validate, the rest train."""
    users = list(range(1, n + 1))
    i = users.index(test_user)
    val = [users[(i + 1 + k) % n] for k in range(3)]
    return val, [u for u in users if u != test_user and u not in val]


def model_mae(name):
    recs = {}
    for f in sorted((RESULTS / name).glob("fold*_testUser*_metrics.json")):
        recs[int(f.name.split("testUser")[1][:2])] = json.loads(f.read_text())["test_mae"]
    return np.array([recs[u] for u in sorted(recs)]), sorted(recs)


# ── computation from the dataset ─────────────────────────────────────────────

def load_participant(user, meta):
    """Return targets in degrees and their sequence-position multiplicities."""
    pu = meta["per_user"][f"User{user}"]
    jm = np.asarray(pu["jnt_mean"], np.float64)
    js = np.asarray(pu["jnt_std"], np.float64)
    ys, ws = [], []
    for seg in sorted((DATA_ROOT / f"User{user}").glob("seg_*.npz")):
        y = np.load(seg)["y"].astype(np.float64) * js + jm
        check_trial(len(y))
        ys.append(y)
        ws.append(window_multiplicity(len(y)))
    return np.concatenate(ys), np.concatenate(ws).astype(np.float64)


def score(y, w, posture):
    e = y - np.asarray(posture, np.float64)[None, :]
    tot = w.sum()
    ss_res = float((w[:, None] * e ** 2).sum())
    mu_j = (w[:, None] * y).sum(axis=0) / tot
    ss_tot = float((w[:, None] * (y - mu_j) ** 2).sum())
    return {"mae_deg": float((w[:, None] * np.abs(e)).sum() / (tot * J)),
            "rmse_deg": float(np.sqrt(ss_res / (tot * J))),
            "r2_joint_centred": 1.0 - ss_res / ss_tot}


def compute(empirical=False):
    meta_path = DATA_ROOT / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(
            f"{meta_path} not found. Writing this record needs the processed dataset, "
            f"which is not distributed; set MYODIGITS_DATA_ROOT, or use --verify for the "
            f"data-free check.")
    meta = json.loads(meta_path.read_text())
    meta_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    own = {u: np.asarray(meta["per_user"][f"User{u}"]["jnt_mean"], np.float64)
           for u in range(1, N_PARTICIPANTS + 1)}

    participants = []
    for u in range(1, N_PARTICIPANTS + 1):
        y, w = load_participant(u, meta)
        val, train = fold_roles(u)
        assert len(train) == 17, train
        assert u not in train, f"User{u}: held-out participant in its own fitting set"
        assert not set(val) & set(train), f"User{u}: validation participant in the fitting set"

        strict_posture = np.mean([own[t] for t in train], axis=0)
        # the same quantity the strict runs recorded, when that fold artifact is published
        for run in ("strict_norm_tx", "strict_norm_gru"):
            f = (RESULTS / run / "strict_norm_stats"
                 / f"fold{u:02d}_testUser{u:02d}_strict_norm_stats.json")
            if not f.is_file():
                continue
            st = json.loads(f.read_text())
            assert st["train_users"] == sorted(train), (run, u, st["train_users"])
            assert st["excluded_users"] == sorted([u] + val), (run, u)
            d = np.abs(np.asarray(st["jnt_mean"], np.float64) - strict_posture).max()
            assert d < 1e-9, f"{run} fold{u:02d}: recorded jnt_mean differs by {d:.3e}"

        rec = {"participant": f"P{u:02d}", "test_user": u,
               "validation_users": val, "n_train_participants": len(train),
               "n_sequence_positions": int(w.sum()),
               "strict_posture": score(y, w, strict_posture),
               "calibrated_posture": score(y, w, own[u])}
        if empirical:
            emp_own = (w[:, None] * y).sum(axis=0) / w.sum()
            rec["empirical_variant"] = {
                "calibrated_posture_mae_deg": score(y, w, emp_own)["mae_deg"],
                "calibrated_posture_r2_joint_centred": score(y, w, emp_own)["r2_joint_centred"],
            }
        participants.append(rec)
        print(f"  P{u:02d}  strict {rec['strict_posture']['mae_deg']:7.4f}   "
              f"calibrated {rec['calibrated_posture']['mae_deg']:7.4f}", flush=True)

    return build_record(participants, meta_sha, empirical)


# ── aggregation, which needs no dataset ──────────────────────────────────────

def aggregate(participants, key):
    v = np.array([p[key]["mae_deg"] for p in participants])
    r = np.array([p[key]["rmse_deg"] for p in participants])
    q = np.array([p[key]["r2_joint_centred"] for p in participants])
    order = [p["participant"] for p in participants]
    return {"mean_mae_deg": float(v.mean()), "sd_mae_deg_ddof0": float(v.std(ddof=0)),
            "mean_rmse_deg": float(r.mean()), "mean_r2_joint_centred": float(q.mean()),
            "best_participant": order[int(v.argmin())], "best_mae_deg": float(v.min()),
            "worst_participant": order[int(v.argmax())], "worst_mae_deg": float(v.max())}


def reductions(participants):
    """Model gain over its protocol-matched posture, both aggregation formulas."""
    base = {"training_statistics_only":
                np.array([p["strict_posture"]["mae_deg"] for p in participants]),
            "participant_calibrated":
                np.array([p["calibrated_posture"]["mae_deg"] for p in participants])}
    rows = []
    for protocol, entries in MODELS.items():
        b = base[protocol]
        for run, label in entries:
            if not (RESULTS / run).is_dir():
                continue
            m, _ = model_mae(run)
            per = (b - m) / b * 100.0
            rows.append({
                "protocol": protocol, "run": run, "model": label,
                "posture_mean_mae_deg": float(b.mean()),
                "model_mean_mae_deg": float(m.mean()),
                "reduction_pct_from_mean_maes": float((b.mean() - m.mean()) / b.mean() * 100.0),
                "reduction_pct_mean_of_per_participant": float(per.mean()),
                "participants_where_the_model_is_worse":
                    [participants[k]["participant"] for k in np.flatnonzero(per < 0)],
            })
    return rows


def build_record(participants, meta_sha, empirical):
    return {
        "description": "Protocol-matched constant-posture baselines, thesis Table 7.4 and "
                       "Table A.4.",
        "generated_by": "scripts/analysis/posture_baselines.py",
        "data_meta_sha256": meta_sha,
        "sequence_length": SEQ_LEN,
        "n_joints": J,
        "fold_rule": "launch.py: fold k tests participant k; the next three participants "
                     "cyclically validate; the remaining 17 train",
        "formulas": {
            "strict_posture": "participant-balanced mean over the 17 training participants "
                              "of that participant's per-joint mean angle, as fitted by "
                              "scripts/normalisation/strict_norm.py",
            "calibrated_posture": "the held-out participant's own per-joint mean angle "
                                  "(meta.json per_user jnt_mean)",
            "scoring": "flat mean over scored rows and 20 joints, rows weighted by "
                       "sequence-position multiplicity "
                       "(myodigits.analysis.weighting.window_multiplicity)",
            "reduction_pct_from_mean_maes":
                "100 * (mean_k posture_k - mean_k model_k) / mean_k posture_k, the form the "
                "thesis prints",
            "reduction_pct_mean_of_per_participant":
                "mean_k [100 * (posture_k - model_k) / posture_k]; a different quantity, "
                "recorded so the two are never interchanged",
        },
        "caveats": {
            "calibrated_posture": "test-label-informed. Fitted on the held-out "
                                  "participant's own reference labels, so it is a "
                                  "diagnostic bound, not a label-free prediction and not a "
                                  "deployable baseline.",
            "posture_statistic": "meta.json per-participant means are over every 2 kHz "
                                 "sample of the continuous angle series, while a scored "
                                 "target is the last sample of its window; the two differ "
                                 "by up to about 0.35 degrees per joint. The thesis uses "
                                 "the documented meta.json statistic.",
            "empirical_variant_present": bool(empirical),
        },
        "participants": participants,
        "aggregate": {
            "strict_posture": aggregate(participants, "strict_posture"),
            "calibrated_posture": aggregate(participants, "calibrated_posture"),
        },
        "reductions": reductions(participants),
    }


# ── data-free verification ───────────────────────────────────────────────────

def verify(record) -> list[str]:
    bad = []
    ps = record["participants"]
    if len(ps) != N_PARTICIPANTS:
        bad.append(f"{len(ps)} participants, expected {N_PARTICIPANTS}")
    for p in ps:
        val, train = fold_roles(p["test_user"])
        if p["validation_users"] != val:
            bad.append(f"{p['participant']}: validation users {p['validation_users']} != {val}")
        if p["n_train_participants"] != 17:
            bad.append(f"{p['participant']}: {p['n_train_participants']} training participants")
        if p["test_user"] in train:
            bad.append(f"{p['participant']}: held-out participant inside its own fitting set")
        if p["n_sequence_positions"] != 66 * (574 - SEQ_LEN + 1) * SEQ_LEN:
            bad.append(f"{p['participant']}: {p['n_sequence_positions']} scored positions")
    for key in ("strict_posture", "calibrated_posture"):
        got, want = aggregate(ps, key), record["aggregate"][key]
        for k, v in want.items():
            if isinstance(v, float):
                if abs(got[k] - v) > 1e-9:
                    bad.append(f"{key}.{k}: record {v!r} != recomputed {got[k]!r}")
            elif got[k] != v:
                bad.append(f"{key}.{k}: record {v!r} != recomputed {got[k]!r}")
    for got, want in zip(reductions(ps), record["reductions"]):
        for k, v in want.items():
            if isinstance(v, float) and abs(got[k] - v) > 1e-9:
                bad.append(f"reduction {want['run']}.{k}: record {v!r} != recomputed {got[k]!r}")
            elif not isinstance(v, float) and got[k] != v:
                bad.append(f"reduction {want['run']}.{k}: record {v!r} != recomputed {got[k]!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="recompute from the dataset and fail if the record is stale")
    g.add_argument("--verify", action="store_true",
                   help="data-free: recompute the aggregates from the committed record")
    ap.add_argument("--empirical", action="store_true",
                    help="also record the windowed-target mean posture as a sensitivity figure")
    args = ap.parse_args()

    if args.verify:
        if not OUT.is_file():
            print(f"MISSING: {OUT.relative_to(REPO)}")
            return 1
        problems = verify(json.loads(OUT.read_text()))
        if problems:
            print("INCONSISTENT:")
            for p in problems:
                print(f"  {p}")
            return 1
        print(f"verified without data: {OUT.relative_to(REPO)}")
        return 0

    record = compute(empirical=args.empirical)
    text = json.dumps(record, indent=1) + "\n"
    if args.check:
        if not OUT.is_file() or OUT.read_text() != text:
            print(f"STALE: {OUT.relative_to(REPO)}")
            return 1
        print(f"up to date: {OUT.relative_to(REPO)}")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    a = record["aggregate"]
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  strict posture      {a['strict_posture']['mean_mae_deg']:.4f} deg")
    print(f"  calibrated posture  {a['calibrated_posture']['mean_mae_deg']:.4f} deg")
    for r in record["reductions"]:
        print(f"  {r['run']:16s} {r['model_mean_mae_deg']:7.4f} deg, "
              f"reduction {r['reduction_pct_from_mean_maes']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
