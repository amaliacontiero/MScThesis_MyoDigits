"""LOSO summary reconstruction from the per-fold metric files on disk.

THE BUG THIS REPLACES
    Up to commit 8dd416b, ``launch.py`` merged previously completed folds into
    the final summary only when an earlier ``loso_summary.json`` already existed
    and the run was resumed with ``start_fold > 1``. The summary is written only
    after the fold loop finishes, so a run killed mid-loop leaves no summary,
    and the next resume silently aggregated only the folds of its own session.
    The strict-normalisation TX run hit this twice (interrupted during fold 2
    and again during fold 9) and wrote a summary of 13 folds, 9 to 21. The
    per-fold records were never affected; only the aggregate was wrong.

THE FIX
    Never read an earlier summary. Rebuild from every ``fold*/metrics.json``,
    validate each record's identity against its directory name, detect missing
    and duplicate folds, label an incomplete run explicitly, and write
    atomically so an interruption cannot leave a partial file.

The aggregation itself (uniform mean over folds, ``numpy.std`` with ddof=0) is
unchanged from the historical summaries, which the regression tests reproduce.
The output is identical to the inline implementation in the ``launch.py``
snapshot used by the strict-normalisation GRU run; ``tests/test_loso_summary.py``
checks that equivalence.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

FOLD_DIR_RE = re.compile(r"^fold(\d+)_testUser(\d+)$")


def collect_fold_metrics(exp_dir):
    """Every valid fold record on disk, with identity validated.

    A record is used only if its directory name parses and any stored ``fold`` /
    ``test_user`` agrees with it. Mismatches and duplicates are reported, never
    silently averaged. Returns ``(records, problems)``.
    """
    records, problems = [], []
    for path in sorted(Path(exp_dir).glob("fold*/metrics.json")):
        d = path.parent.name
        m = FOLD_DIR_RE.match(d)
        if not m:
            problems.append(f"unparsable fold directory: {d}")
            continue
        dir_fold, dir_user = int(m.group(1)), int(m.group(2))
        try:
            j = json.loads(path.read_text())
        except Exception as e:
            problems.append(f"{d}: unreadable metrics.json ({type(e).__name__})")
            continue
        if "test_mae" not in j:
            problems.append(f"{d}: metrics.json has no test_mae")
            continue
        sf, su = j.get("fold"), j.get("test_user")
        if sf is not None and sf != dir_fold:
            problems.append(f"{d}: stored fold {sf} != directory fold {dir_fold}")
            continue
        if su is not None and su != dir_user:
            problems.append(f"{d}: stored test_user {su} != directory user {dir_user}")
            continue
        j.setdefault("fold", dir_fold)
        j.setdefault("test_user", dir_user)
        records.append(j)

    folds = [r["fold"] for r in records]
    users = [r["test_user"] for r in records]
    if len(set(folds)) != len(folds):
        problems.append(f"duplicate fold numbers: {sorted({f for f in folds if folds.count(f) > 1})}")
    if len(set(users)) != len(users):
        problems.append(f"duplicate test users: {sorted({u for u in users if users.count(u) > 1})}")

    records.sort(key=lambda r: r["fold"])
    return records, problems


def expected_folds_for(n_users, proxy=False, proxy_folds=None, all_users=None):
    """The ``expected_folds`` argument ``launch.py`` passes for a run.

    Full LOSO expects folds 1..n_users. A proxy run expects exactly the fold
    numbers of its listed test users, whatever order they were run in.
    """
    if proxy:
        users = list(all_users) if all_users is not None else list(range(1, n_users + 1))
        return sorted(users.index(u) + 1 for u in proxy_folds)
    return int(n_users)


def build_summary(exp_dir, run_name, expected_folds=None):
    """Reconstruct the summary dictionary from disk. Never reads a previous summary.

    expected_folds
        int              a full run that must cover folds 1..N
        iterable of int  a proxy run that must cover exactly these fold numbers
        None             coverage is recorded but not asserted
    """
    all_fold_metrics, problems = collect_fold_metrics(exp_dir)
    if not all_fold_metrics:
        raise ValueError(f"no valid fold metrics found under {exp_dir}")

    maes = [m["test_mae"] for m in all_fold_metrics]
    rmses = [m["test_rmse"] for m in all_fold_metrics]
    r2s = [m["test_r2"] for m in all_fold_metrics]
    mean_rs = [m["mean_pearson_r"] for m in all_fold_metrics]

    found = sorted(m["fold"] for m in all_fold_metrics)
    complete, expected_desc, missing = None, None, []
    if isinstance(expected_folds, (int, np.integer)) and not isinstance(expected_folds, bool):
        expected_desc = f"folds 1-{expected_folds}"
        missing = sorted(set(range(1, int(expected_folds) + 1)) - set(found))
        complete = not missing
    elif expected_folds is not None:
        want = sorted(expected_folds)
        expected_desc = f"folds {want}"
        missing = sorted(set(want) - set(found))
        complete = not missing
    if missing:
        problems.append(f"missing folds: {missing}")

    summary = {
        "run": run_name,
        "n_folds": len(all_fold_metrics),
        "mean_test_mae": float(np.mean(maes)),
        "std_test_mae": float(np.std(maes)),
        "mean_test_rmse": float(np.mean(rmses)),
        "std_test_rmse": float(np.std(rmses)),
        "mean_test_r2": float(np.mean(r2s)),
        "std_test_r2": float(np.std(r2s)),
        "mean_pearson_r": float(np.mean(mean_rs)),
        "std_pearson_r": float(np.std(mean_rs)),
        "per_fold": all_fold_metrics,
        "aggregation": {
            "source": "reconstructed from fold*/metrics.json on disk",
            "reads_previous_summary": False,
            "sd_convention": "ddof=0 (numpy default)",
            "folds_found": found,
            "expected": expected_desc,
            "missing_folds": missing,
            "complete": complete,
            "problems": problems,
        },
    }
    if complete is False:
        summary["INCOMPLETE"] = True
        summary["incomplete_reason"] = f"missing folds {missing}"
    return summary


def write_summary_atomic(summary, exp_dir, filename="loso_summary.json"):
    """Write through a temporary file and ``os.replace``; no partial file is ever visible."""
    summary_path = Path(exp_dir) / filename
    fd, tmp = tempfile.mkstemp(dir=str(exp_dir), prefix=".loso_summary_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(summary, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, str(summary_path))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return summary_path


def save_loso_summary(exp_dir, run_name, expected_folds=None, filename="loso_summary.json"):
    """Rebuild, write atomically and print the summary. Returns the summary dict."""
    summary = build_summary(exp_dir, run_name, expected_folds)
    summary_path = write_summary_atomic(summary, exp_dir, filename)

    agg = summary["aggregation"]
    if agg["problems"]:
        print("\n  SUMMARY PROBLEMS:")
        for x in agg["problems"]:
            print(f"    - {x}")
    if agg["complete"] is False:
        print(f"  *** RUN INCOMPLETE: missing folds {agg['missing_folds']} ***")

    print(f"\n{'='*60}")
    print(f"LOSO CV Summary — {run_name}")
    print(f"{'='*60}")
    print(f"MAE:       {summary['mean_test_mae']:.4f} ± {summary['std_test_mae']:.4f}")
    print(f"RMSE:      {summary['mean_test_rmse']:.4f} ± {summary['std_test_rmse']:.4f}")
    print(f"R²:        {summary['mean_test_r2']:.4f} ± {summary['std_test_r2']:.4f}  "
          f"(globally pooled; see RESULTS.md)")
    print(f"Pearson r: {summary['mean_pearson_r']:.4f} ± {summary['std_pearson_r']:.4f}")
    print(f"\nSaved: {summary_path}")
    return summary


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m myodigits.analysis.loso_summary",
        description="Rebuild a LOSO summary from the per-fold metrics.json files of a run directory.")
    ap.add_argument("run_dir")
    ap.add_argument("--expected-folds", type=int, default=21,
                    help="number of folds a complete run has (default 21)")
    ap.add_argument("--proxy-folds", default=None,
                    help="comma-separated fold numbers for a proxy run, instead of --expected-folds")
    ap.add_argument("--write", action="store_true",
                    help="write loso_summary.json atomically; without it, only print the result")
    args = ap.parse_args(argv)
    expected = ([int(x) for x in args.proxy_folds.split(",") if x.strip()]
                if args.proxy_folds else args.expected_folds)
    run_dir = Path(args.run_dir)
    if args.write:
        s = save_loso_summary(run_dir, run_dir.name, expected_folds=expected)
    else:
        s = build_summary(run_dir, run_dir.name, expected_folds=expected)
        a = s["aggregation"]
        print(f"{run_dir.name}: n_folds={s['n_folds']} complete={a['complete']} "
              f"MAE {s['mean_test_mae']:.4f} ± {s['std_test_mae']:.4f} (ddof=0)")
        for p in a["problems"]:
            print(f"  problem: {p}")
    return 0 if s["aggregation"]["complete"] is not False and not s["aggregation"]["problems"] else 1


def materialise_release_layout(results_dir, target_dir):
    """Recreate ``fold*/metrics.json`` run layout from the flat release layout.

    The release stores ``foldNN_testUserNN_metrics.json`` side by side. This
    copies them into ``target_dir/foldNN_testUserNN/metrics.json`` so the same
    reconstruction code runs on published results.
    """
    results_dir, target_dir = Path(results_dir), Path(target_dir)
    n = 0
    for f in sorted(results_dir.glob("fold*_testUser*_metrics.json")):
        name = f.name[: -len("_metrics.json")]
        d = target_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_bytes(f.read_bytes())
        n += 1
    return n


if __name__ == "__main__":
    raise SystemExit(_cli())
