#!/usr/bin/env python3
"""Extract per-joint R² sufficient statistics from cached test predictions.

REQUIRES PRIVATE INPUTS. Needs one ``UNN.npz`` per held-out participant, each
holding ``pred`` and ``targ`` arrays of shape (n_windows, 20) in degrees. Those
come from evaluating the trained checkpoints on PiMForce and are not
distributed. Everyone else uses the published output of this script through
``r2_reconciliation.py``.

For every fold the script validates, before writing anything, that the arrays
belong to the published fold record:

  * row count equals ``n_test_windows``;
  * MAE recomputed from the arrays matches ``test_mae``;
  * RMSE implied by the statistics matches ``test_rmse``;
  * the globally pooled variant matches the stored ``test_r2`` (eval.py stores
    exactly that variant, in float32).

Usage
    python scripts/analysis/extract_r2_statistics.py \\
        --config myodigits_tx --preds-dir PATH/TO/preds --out results/analysis/r2
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from myodigits.analysis import r2_variants as r2  # noqa: E402

# float32 accumulation over ~43 million terms in eval.py bounds how closely a
# float64 recomputation can agree with the stored values.
TOL_MAE = 2e-3
TOL_RMSE = 2e-3
TOL_STORED_R2 = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    help="results/<config> directory holding the per-fold metrics")
    ap.add_argument("--preds-dir", required=True, type=Path,
                    help="directory of UNN.npz prediction files (not distributed)")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "analysis" / "r2")
    ap.add_argument("--n-folds", type=int, default=21)
    args = ap.parse_args()

    metrics_dir = REPO / "results" / args.config
    folds, problems = [], []
    for k in range(1, args.n_folds + 1):
        mpath = metrics_dir / f"fold{k:02d}_testUser{k:02d}_metrics.json"
        npz = args.preds_dir / f"U{k:02d}.npz"
        if not mpath.exists():
            problems.append(f"fold {k}: metrics file missing")
            continue
        if not npz.exists():
            problems.append(f"fold {k}: prediction file missing")
            continue
        m = json.loads(mpath.read_text())
        z = np.load(npz)
        P, T = z["pred"], z["targ"]
        if P.shape != T.shape or P.shape[1] != 20:
            problems.append(f"fold {k}: bad shapes {P.shape} / {T.shape}")
            continue
        if P.shape[0] != m["n_test_windows"]:
            problems.append(f"fold {k}: {P.shape[0]} rows vs n_test_windows {m['n_test_windows']}")
            continue

        abs_sum = 0.0
        for s in range(0, len(T), 250_000):
            abs_sum += float(np.abs(T[s:s + 250_000].astype(np.float64)
                                    - P[s:s + 250_000].astype(np.float64)).sum())
        mae = abs_sum / T.size

        stats = r2.sufficient_statistics(P, T)
        values = r2.r2_from_statistics(stats)
        d_mae = abs(mae - m["test_mae"])
        d_rmse = r2.rmse_discrepancy(stats, m["test_rmse"])
        d_r2 = abs(values[r2.GLOBALLY_POOLED] - m["test_r2"])
        ok = d_mae <= TOL_MAE and d_rmse <= TOL_RMSE and d_r2 <= TOL_STORED_R2
        if not ok:
            problems.append(f"fold {k}: validation failed dMAE={d_mae:.2e} "
                            f"dRMSE={d_rmse:.2e} dR2stored={d_r2:.2e}")
            continue
        folds.append({
            "fold": k, "test_user": k,
            **stats,
            "validation": {"abs_diff_mae": d_mae, "abs_diff_rmse": d_rmse,
                           "abs_diff_stored_test_r2": d_r2},
            **values,
        })
        print(f"  fold {k:2d}  dMAE {d_mae:.1e}  dRMSE {d_rmse:.1e}  dR2stored {d_r2:.1e}")

    if problems or len(folds) != args.n_folds:
        print("NOT WRITTEN, problems:")
        for p in problems:
            print("   ", p)
        return 1

    out = {
        "config": args.config,
        "description": "per-joint sufficient statistics for the three R² variants, "
                       "one record per held-out participant",
        "definitions": {v: r2.LABELS[v] for v in r2.VARIANTS},
        "n_folds": len(folds),
        "tolerances": {"mae": TOL_MAE, "rmse": TOL_RMSE, "stored_test_r2": TOL_STORED_R2},
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.system()},
        "folds": folds,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.config}.json"
    target.write_text(json.dumps(out, indent=1))
    print(f"wrote {target.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
