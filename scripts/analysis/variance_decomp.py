"""
Cross-subject variance analysis for run06 (or any LOSO run with per-fold metrics).

Quantifies the inter-subject performance gap that motivates RQ4.

Outputs:
  - per-subject MAE distribution stats
  - per-finger MAE breakdown by subject
  - rank correlation between subjects' MAE and per-joint difficulty
  - figures (CDF, per-subject bar with error, finger heatmap)

This is empirical contribution #1 of the paper: showing the gap is not just
mean ± std, but is structurally driven by which subjects have which kinds of
muscle activation.

Usage:
  python scripts/variance_decomp.py
  python scripts/variance_decomp.py --run run06_posenet_seq_lr3e-4_bs64_ep60_loso
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

JOINT_NAMES = [
    "thumb-cmc-spread",  "thumb-cmc-stretch",  "thumb-mcp-stretch", "thumb-ip-stretch",
    "index-mcp-spread",  "index-mcp-stretch",  "index-pip-stretch", "index-dip-stretch",
    "middle-mcp-spread", "middle-mcp-stretch", "middle-pip-stretch","middle-dip-stretch",
    "ring-mcp-spread",   "ring-mcp-stretch",   "ring-pip-stretch",  "ring-dip-stretch",
    "pinky-mcp-spread",  "pinky-mcp-stretch",  "pinky-pip-stretch", "pinky-dip-stretch",
]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def _load_per_fold(run_dir):
    """Read each fold's metrics.json and return a list of fold dicts."""
    folds = []
    for fold_dir in sorted(run_dir.glob("fold*_testUser*")):
        m_path = fold_dir / "metrics.json"
        if not m_path.exists():
            continue
        m = json.loads(m_path.read_text())
        # parse user id
        try:
            user_id = int(fold_dir.name.split("testUser")[-1])
        except ValueError:
            continue
        m["test_user"] = user_id
        folds.append(m)
    return sorted(folds, key=lambda x: x["test_user"])


def _finger_mae(per_joint_mae_dict):
    """Aggregate 20-joint MAE dict into 5-finger means."""
    out = {}
    for f in FINGERS:
        vals = [v for k, v in per_joint_mae_dict.items() if k.startswith(f)]
        out[f] = float(np.mean(vals)) if vals else float("nan")
    return out


def _summary_stats(maes):
    return {
        "n":     len(maes),
        "mean":  float(np.mean(maes)),
        "std":   float(np.std(maes)),
        "min":   float(np.min(maes)),
        "max":   float(np.max(maes)),
        "p25":   float(np.percentile(maes, 25)),
        "p50":   float(np.percentile(maes, 50)),
        "p75":   float(np.percentile(maes, 75)),
        "iqr":   float(np.percentile(maes, 75) - np.percentile(maes, 25)),
        "cv":    float(np.std(maes) / np.mean(maes)),     # coefficient of variation
        "range_ratio": float(np.max(maes) / np.min(maes)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run06_posenet_seq_lr3e-4_bs64_ep60_loso")
    parser.add_argument("--out", default=None,
                        help="Output dir (default: <run>/variance_analysis)")
    args = parser.parse_args()

    run_dir = ROOT / "experiments" / args.run
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = Path(args.out) if args.out else (run_dir / "variance_analysis")
    out_dir.mkdir(exist_ok=True)

    print(f"Loading per-fold metrics from {run_dir}")
    folds = _load_per_fold(run_dir)
    if not folds:
        raise RuntimeError("No fold metrics.json files found.")
    print(f"Loaded {len(folds)} folds.\n")

    # ── per-subject MAE distribution ─────────────────────────────────────────
    user_ids   = [f["test_user"]  for f in folds]
    test_maes  = [f["test_mae"]   for f in folds]
    test_r2s   = [f["test_r2"]    for f in folds]
    pearson_rs = [f["mean_pearson_r"] for f in folds]

    stats = _summary_stats(test_maes)
    print("=" * 60)
    print("Cross-subject MAE distribution")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<14s} {v:.4f}")
    print()

    # ── rank subjects ────────────────────────────────────────────────────────
    ranked = sorted(zip(user_ids, test_maes), key=lambda x: x[1])
    print("Rank | User | MAE")
    for rank, (u, m) in enumerate(ranked, 1):
        marker = ""
        if rank == 1:                marker = "  ← BEST"
        if rank == len(ranked):      marker = "  ← WORST"
        if rank == len(ranked) // 2: marker = "  ← MEDIAN"
        print(f"{rank:>3d}. | User{u:02d} | {m:6.3f}{marker}")
    print()

    # ── per-finger breakdown ─────────────────────────────────────────────────
    finger_table = {}     # user -> {finger: MAE}
    for f in folds:
        finger_table[f["test_user"]] = _finger_mae(f["per_joint_mae"])

    finger_arr = np.array([
        [finger_table[u][fg] for fg in FINGERS]
        for u in user_ids
    ])  # (n_users, 5)

    print("Per-finger mean MAE across subjects")
    print("-" * 40)
    for j, fg in enumerate(FINGERS):
        col = finger_arr[:, j]
        print(f"  {fg:<8s}  mean={col.mean():.3f}  std={col.std():.3f}  "
              f"range=[{col.min():.3f}, {col.max():.3f}]")
    print()

    # ── figure 1: MAE bar chart per subject (sorted) ─────────────────────────
    sorted_idx = np.argsort(test_maes)
    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(range(len(user_ids)),
                  [test_maes[i] for i in sorted_idx],
                  color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(user_ids)))
    ax.set_xticklabels([f"U{user_ids[i]:02d}" for i in sorted_idx],
                       rotation=45, fontsize=8)
    ax.axhline(stats["mean"], color="red", ls="--",
               label=f"mean = {stats['mean']:.3f}")
    ax.fill_between([-0.5, len(user_ids)-0.5],
                    stats["mean"] - stats["std"],
                    stats["mean"] + stats["std"],
                    color="red", alpha=0.10, label=f"±1σ ({stats['std']:.3f})")
    ax.set_ylabel("Test MAE (°)")
    ax.set_title(f"{args.run} — Per-subject MAE  "
                 f"(range={stats['min']:.2f}–{stats['max']:.2f}°, "
                 f"ratio={stats['range_ratio']:.2f}×)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_subject_mae.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir/'per_subject_mae.png'}")

    # ── figure 2: per-finger heatmap (subject × finger) ──────────────────────
    fig, ax = plt.subplots(figsize=(7, 8))
    order = np.argsort(test_maes)
    im = ax.imshow(finger_arr[order], aspect="auto", cmap="RdYlGn_r",
                   vmin=np.percentile(finger_arr, 5),
                   vmax=np.percentile(finger_arr, 95))
    ax.set_xticks(range(5)); ax.set_xticklabels(FINGERS)
    ax.set_yticks(range(len(user_ids)))
    ax.set_yticklabels([f"U{user_ids[i]:02d} ({test_maes[i]:.2f})" for i in order],
                       fontsize=8)
    ax.set_title(f"{args.run} — Per-finger MAE (subjects sorted by overall MAE)")
    fig.colorbar(im, ax=ax, label="MAE (°)")
    fig.tight_layout()
    fig.savefig(out_dir / "finger_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir/'finger_heatmap.png'}")

    # ── figure 3: CDF of subject MAE ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    s = np.sort(test_maes)
    cdf = np.arange(1, len(s) + 1) / len(s)
    ax.plot(s, cdf, "o-", lw=2, color="steelblue")
    ax.axvline(stats["mean"],  ls="--", color="red", label=f"mean={stats['mean']:.2f}")
    ax.axvline(stats["p50"],   ls=":",  color="green", label=f"median={stats['p50']:.2f}")
    ax.set_xlabel("Subject MAE (°)")
    ax.set_ylabel("Cumulative fraction of subjects")
    ax.set_title("CDF of cross-subject MAE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mae_cdf.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir/'mae_cdf.png'}")

    # ── persist stats ────────────────────────────────────────────────────────
    summary = {
        "run":                args.run,
        "n_subjects":         len(user_ids),
        "summary_stats":      stats,
        "per_subject_mae":    {f"User{u:02d}": m for u, m in zip(user_ids, test_maes)},
        "per_subject_r2":     {f"User{u:02d}": r for u, r in zip(user_ids, test_r2s)},
        "per_subject_pearson":{f"User{u:02d}": r for u, r in zip(user_ids, pearson_rs)},
        "per_finger_mae":     {f"User{u:02d}": finger_table[u] for u in user_ids},
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"\nKey numbers for the paper:")
    print(f"  Best subject  (User{ranked[0][0]:02d}):    MAE = {ranked[0][1]:.3f}°")
    print(f"  Worst subject (User{ranked[-1][0]:02d}):    MAE = {ranked[-1][1]:.3f}°")
    print(f"  Worst-to-best ratio:        {stats['range_ratio']:.2f}×")
    print(f"  CV (std / mean):            {stats['cv']:.3f}")
    print(f"  IQR:                        {stats['iqr']:.3f}°")


if __name__ == "__main__":
    main()
