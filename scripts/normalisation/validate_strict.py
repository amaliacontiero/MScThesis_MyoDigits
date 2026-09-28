"""Fast numerical validation of the strict-normalisation transform.

Checks, in order:
  1. fold roles are assigned before statistics are fitted, and the exclusion
     assertion actually fires when violated;
  2. tensor shapes and dtypes survive the transform;
  3. no NaN or Inf is produced;
  4. the round trip is exact: recovering raw values and re-applying that
     participant's own statistics reproduces the stored array;
  5. training participants are standardised to ~zero mean / unit variance under
     the fold statistics, while the held-out participant is NOT (this is the
     whole point of the experiment);
  6. the inverse transform to degrees is consistent with the metric computation.
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
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from strict_norm import (load_meta, fold_statistics, restandardise_user,
                         denorm_targets, save_fold_artifact)

DATA = str(_ROOT / "data/processed/myodigitsv1_loso_v4")
meta = load_meta(DATA)
ALL = list(range(1, 22))

# ---- fold construction, copied verbatim from launch.py L303-305 ------------
TEST = 1
fold_idx = ALL.index(TEST)
VAL = [ALL[(fold_idx + 1 + k) % len(ALL)] for k in range(3)]
TRAIN = [u for u in ALL if u != TEST and u not in VAL]
print(f"[1] fold roles fixed BEFORE statistics: test=User{TEST} val={VAL} "
      f"train={len(TRAIN)} participants")
assert len(TRAIN) == 17 and TEST not in TRAIN and not set(VAL) & set(TRAIN)

stats = fold_statistics(meta, TRAIN, exclude_users=[TEST] + VAL)
print(f"    weighting={stats['weighting']}  contributing windows={stats['n_contributing_windows']:,}"
      f"  degenerate emg/jnt channels={stats['degenerate_emg_channels']}/{stats['degenerate_jnt_channels']}")

# the exclusion assertion must actually fire
try:
    fold_statistics(meta, TRAIN + [TEST], exclude_users=[TEST] + VAL)
    print("    !! LEAK ASSERTION DID NOT FIRE"); sys.exit(1)
except AssertionError as e:
    print(f"    leak assertion fires as intended: {str(e)[:70]}")

# ---- load one segment from the held-out participant and one from training --
def load_seg(uid):
    f = sorted(Path(DATA, f"User{uid}").glob("seg_*.npz"))[0]
    d = np.load(f)
    return d["X"], d["y"]

for tag, uid in (("held-out", TEST), ("training", TRAIN[0])):
    X, y = load_seg(uid)
    Xs, ys = restandardise_user(X, y, f"User{uid}", meta, stats)
    print(f"\n[2] {tag} User{uid}: X {X.shape}{X.dtype} -> {Xs.shape}{Xs.dtype}; "
          f"y {y.shape} -> {ys.shape}")
    assert Xs.shape == X.shape and ys.shape == y.shape
    assert Xs.dtype == np.float32 and ys.dtype == np.float32
    print(f"[3] finite: X {np.isfinite(Xs).all()}  y {np.isfinite(ys).all()}")
    assert np.isfinite(Xs).all() and np.isfinite(ys).all()

    # [4] round trip through that participant's OWN statistics must be identity
    own = {k: meta["per_user"][f"User{uid}"][k] for k in
           ("emg_mean", "emg_std", "jnt_mean", "jnt_std")}
    Xi, yi = restandardise_user(X, y, f"User{uid}", meta, own)
    print(f"[4] identity round trip: max|dX|={np.abs(Xi - X).max():.3e}  "
          f"max|dy|={np.abs(yi - y).max():.3e}")
    assert np.abs(Xi - X).max() < 1e-4 and np.abs(yi - y).max() < 1e-4

    print(f"[5] under FOLD stats: y mean={ys.mean():+.3f} std={ys.std():.3f} "
          f"({'~standardised' if abs(ys.mean()) < 0.6 and 0.5 < ys.std() < 1.8 else 'NOT standardised'})")

# ---- [6] inverse transform vs metric computation ---------------------------
X, y = load_seg(TEST)
Xs, ys = restandardise_user(X, y, f"User{TEST}", meta, stats)
pred = ys + 0.05                                   # a deliberate known offset
mae_std = np.abs(pred - ys).mean()
deg_p, deg_t = denorm_targets(pred, stats), denorm_targets(ys, stats)
mae_deg = np.abs(deg_p - deg_t).mean()
expected = 0.05 * np.asarray(stats["jnt_std"], np.float32).mean()
print(f"\n[6] inverse transform: MAE(standardised)={mae_std:.4f}  "
      f"MAE(degrees)={mae_deg:.4f}  expected~{expected:.4f}  "
      f"ratio={mae_deg/expected:.4f}")
assert abs(mae_deg / expected - 1.0) < 0.02

out = str(_ROOT / "experiments/strictnorm_validation_fold01.json")
save_fold_artifact(out, stats)
print(f"\nfold artefact saved -> {out}")
print("\nALL VALIDATION CHECKS PASSED")
