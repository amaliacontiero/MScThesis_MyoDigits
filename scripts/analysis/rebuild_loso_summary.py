#!/usr/bin/env python3
"""Standalone LOSO summary rebuilder.

Reconstructs loso_summary.json from the individual per-fold metrics.json files.
Needed because launch.py only merges previously completed folds when
start_fold > 1; a run resumed with start_fold=1 skips finished folds via the
metrics.json guard but does not add them back, so the summary it writes at the
end would contain only the folds executed in that final invocation.

This utility touches nothing in the training or evaluation pipeline. By default
it writes loso_summary_rebuilt.json and leaves any existing summary alone.

Usage
    python rebuild_loso_summary.py EXP_DIR                 # rebuild + compare
    python rebuild_loso_summary.py EXP_DIR --write         # overwrite loso_summary.json
    python rebuild_loso_summary.py EXP_DIR --n-users 21

Aggregation matches launch._save_loso_summary exactly, including np.std with
ddof=0 (population standard deviation).
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np

FOLD_RE = re.compile(r"^fold(\d+)_testUser(\d+)$")


def fold_roles(test_user, all_users):
    """Identical to launch.py L314-315."""
    n = len(all_users)
    i = all_users.index(test_user)
    val_users = [all_users[(i + 1 + k) % n] for k in range(3)]
    train_users = [u for u in all_users if u != test_user and u not in val_users]
    return val_users, train_users


def collect(exp_dir, n_users):
    all_users = list(range(1, n_users + 1))
    rows, missing = [], []
    for d in sorted(exp_dir.glob("fold*_testUser*")):
        m = FOLD_RE.match(d.name)
        if not m:
            continue
        fold_no, test_user = int(m.group(1)), int(m.group(2))
        mj = d / "metrics.json"
        if not mj.exists():
            missing.append(d.name)
            continue
        rec = json.loads(mj.read_text())
        rec["fold"] = fold_no
        rec["test_user"] = test_user
        rec["val_users"], _ = fold_roles(test_user, all_users)
        rows.append(rec)
    rows.sort(key=lambda r: r["fold"])
    return rows, missing


def summarise(rows, run_name):
    def agg(key):
        vals = [r[key] for r in rows if key in r]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, None, 0)

    out = {"run": run_name, "n_folds": len(rows)}
    for key, mean_name, std_name in (
        ("test_mae", "mean_test_mae", "std_test_mae"),
        ("test_rmse", "mean_test_rmse", "std_test_rmse"),
        ("test_r2", "mean_test_r2", "std_test_r2"),
        ("mean_pearson_r", "mean_pearson_r", "std_pearson_r"),
    ):
        mu, sd, n = agg(key)
        if mu is not None:
            out[mean_name], out[std_name] = mu, sd
    # extra metrics that only newer eval.py emits; reported but kept separate so
    # the core schema stays byte-comparable with launch.py's own output
    extra = {}
    for key in ("test_r2_pooled",):
        mu, sd, n = agg(key)
        if mu is not None:
            extra[key] = {"mean": mu, "std": sd, "n_folds_present": n}
    if extra:
        out["additional_metrics"] = extra
    out["per_fold"] = rows
    return out


def compare(rebuilt, existing):
    print("\n=== comparison with existing loso_summary.json ===")
    keys = [k for k in existing if k != "per_fold"]
    worst = 0.0
    for k in keys:
        a, b = existing.get(k), rebuilt.get(k)
        if isinstance(a, float) and isinstance(b, float):
            d = abs(a - b)
            worst = max(worst, d)
            flag = "OK" if d < 1e-9 else "DIFFERS"
            print(f"  {k:18s} existing={a:<12.6f} rebuilt={b:<12.6f} diff={d:.2e}  {flag}")
        elif a != b:
            print(f"  {k:18s} existing={a!r} rebuilt={b!r}  DIFFERS")
        else:
            print(f"  {k:18s} {a!r}  OK")

    ex, rb = existing.get("per_fold", []), rebuilt.get("per_fold", [])
    print(f"\n  per_fold length: existing={len(ex)} rebuilt={len(rb)} "
          f"{'OK' if len(ex) == len(rb) else 'DIFFERS'}")
    common = 0
    for a, b in zip(sorted(ex, key=lambda r: r["fold"]), sorted(rb, key=lambda r: r["fold"])):
        for k in ("test_mae", "test_rmse", "test_r2", "mean_pearson_r", "test_user", "fold"):
            if k in a and k in b:
                if isinstance(a[k], float):
                    d = abs(a[k] - b[k])
                    worst = max(worst, d)
                    if d > 1e-9:
                        print(f"    fold {a['fold']:02d} {k}: {a[k]} vs {b[k]}  DIFFERS")
                elif a[k] != b[k]:
                    print(f"    fold {a['fold']:02d} {k}: {a[k]} vs {b[k]}  DIFFERS")
                common += 1
    only_rebuilt = set(rb[0]) - set(ex[0]) if ex and rb else set()
    only_existing = set(ex[0]) - set(rb[0]) if ex and rb else set()
    print(f"  per-fold scalar comparisons made: {common}")
    print(f"  worst numeric difference anywhere: {worst:.2e}")
    if only_rebuilt:
        print(f"  fields present only in rebuilt : {sorted(only_rebuilt)}")
    if only_existing:
        print(f"  fields present only in existing: {sorted(only_existing)}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--n-users", type=int, default=21)
    ap.add_argument("--write", action="store_true",
                    help="overwrite loso_summary.json instead of writing loso_summary_rebuilt.json")
    a = ap.parse_args()

    exp = Path(a.exp_dir)
    if not exp.is_dir():
        sys.exit(f"not a directory: {exp}")

    rows, missing = collect(exp, a.n_users)
    print(f"run directory : {exp.name}")
    print(f"folds found   : {len(rows)}")
    if missing:
        print(f"INCOMPLETE (no metrics.json): {missing}")
    if not rows:
        sys.exit("no completed folds found")

    got = sorted(r["fold"] for r in rows)
    expected = list(range(1, a.n_users + 1))
    gaps = [f for f in expected if f not in got]
    print(f"fold numbers  : {got}")
    if gaps:
        print(f"MISSING FOLDS : {gaps}")

    metric_keys = ("test_mae", "test_rmse", "test_r2", "mean_pearson_r",
                   "test_r2_pooled", "per_joint_mae", "per_joint_r", "per_joint_r2")
    print("\nmetric availability across folds:")
    for k in metric_keys:
        n = sum(1 for r in rows if k in r)
        print(f"  {k:16s} present in {n}/{len(rows)} folds")

    rebuilt = summarise(rows, exp.name)
    print(f"\nrebuilt: MAE {rebuilt['mean_test_mae']:.4f} +/- {rebuilt['std_test_mae']:.4f}"
          f" | RMSE {rebuilt['mean_test_rmse']:.4f} +/- {rebuilt['std_test_rmse']:.4f}"
          f" | R2 {rebuilt['mean_test_r2']:.4f} | r {rebuilt['mean_pearson_r']:.4f}")

    existing_path = exp / "loso_summary.json"
    if existing_path.exists():
        compare(rebuilt, json.loads(existing_path.read_text()))

    out = existing_path if a.write else exp / "loso_summary_rebuilt.json"
    out.write_text(json.dumps(rebuilt, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
