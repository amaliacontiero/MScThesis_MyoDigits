"""
Generate a detailed LOSO CV report notebook for run06.

Usage (from project root):
    python scripts/loso_report_detailed.py
    python scripts/loso_report_detailed.py experiments/run06_...
"""

import io
import sys
import json
import base64
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[2]

JOINT_NAMES = [
    "thumb-cmc-spread",  "thumb-cmc-stretch",  "thumb-mcp-stretch",  "thumb-ip-stretch",
    "index-mcp-spread",  "index-mcp-stretch",  "index-pip-stretch",  "index-dip-stretch",
    "middle-mcp-spread", "middle-mcp-stretch",  "middle-pip-stretch", "middle-dip-stretch",
    "ring-mcp-spread",   "ring-mcp-stretch",   "ring-pip-stretch",   "ring-dip-stretch",
    "pinky-mcp-spread",  "pinky-mcp-stretch",  "pinky-pip-stretch",  "pinky-dip-stretch",
]

FINGER_GROUPS = {
    "thumb":  (0, 4),
    "index":  (4, 8),
    "middle": (8, 12),
    "ring":   (12, 16),
    "pinky":  (16, 20),
}

FINGER_COLORS = {
    "thumb":  "#a78bfa",
    "index":  "#60a5fa",
    "middle": "#34d399",
    "ring":   "#f87171",
    "pinky":  "#fb923c",
}

JOINT_SHORT = [
    "T-CMC-spr", "T-CMC-str", "T-MCP-str", "T-IP-str",
    "I-MCP-spr", "I-MCP-str", "I-PIP-str", "I-DIP-str",
    "M-MCP-spr", "M-MCP-str", "M-PIP-str", "M-DIP-str",
    "R-MCP-spr", "R-MCP-str", "R-PIP-str", "R-DIP-str",
    "P-MCP-spr", "P-MCP-str", "P-PIP-str", "P-DIP-str",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_cell(b64, title=""):
    cell = new_code_cell(source=f"# {title}" if title else "# figure")
    cell.outputs = [nbformat.v4.new_output(
        output_type="display_data",
        data={"image/png": b64, "text/plain": ["<Figure>"]},
        metadata={},
    )]
    cell.metadata["tags"] = ["hide-input"]
    return cell


def _load_history(fold_dir):
    h = fold_dir / "checkpoints" / "history.json"
    return json.loads(h.read_text()) if h.exists() else []


def _finger_for_joint(j):
    for name, (lo, hi) in FINGER_GROUPS.items():
        if lo <= j < hi:
            return name
    return "unknown"


# ── plot helpers ──────────────────────────────────────────────────────────────

def _joint_colors():
    colors = []
    for j in range(20):
        finger = _finger_for_joint(j)
        colors.append(FINGER_COLORS[finger])
    return colors


# ── section 1: overall metrics ────────────────────────────────────────────────

def _plot_per_subject_bar(folds, mean_mae):
    users = [f["test_user"] for f in folds]
    maes  = [f["test_mae"]  for f in folds]
    order = np.argsort(maes)

    cmap  = plt.cm.RdYlGn_r
    norm  = mcolors.Normalize(vmin=min(maes), vmax=max(maes))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [cmap(norm(maes[i])) for i in order]
    bars = ax.barh(
        [f"User{users[i]:02d}" for i in order],
        [maes[i] for i in order],
        color=colors, edgecolor="white", linewidth=0.4, height=0.7,
    )
    ax.axvline(mean_mae, color="#333333", linestyle="--", linewidth=1.5,
               label=f"mean = {mean_mae:.2f}°")
    for bar, i in zip(bars, order):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{maes[i]:.2f}°", va="center", fontsize=8.5)
    ax.set_xlabel("Test MAE (°)", fontsize=11)
    ax.set_title("Per-subject MAE — sorted, 21-fold LOSO", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="MAE (°)", shrink=0.6)
    fig.tight_layout()
    return fig


def _plot_three_metrics_scatter(folds):
    users   = [f["test_user"]      for f in folds]
    maes    = [f["test_mae"]       for f in folds]
    r2s     = [f["test_r2"]        for f in folds]
    mean_rs = [f["mean_pearson_r"] for f in folds]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    data_pairs = [
        (maes,    "MAE (°)",      "steelblue"),
        (r2s,     "R²",           "coral"),
        (mean_rs, "Mean Pearson r", "mediumseagreen"),
    ]
    for ax, (vals, label, color) in zip(axes, data_pairs):
        ax.scatter(users, vals, color=color, s=55, zorder=3, edgecolors="white", linewidths=0.5)
        mean_v = np.mean(vals)
        ax.axhline(mean_v, color="#333", linestyle="--", linewidth=1,
                   label=f"μ={mean_v:.3f}")
        for u, v in zip(users, vals):
            ax.annotate(f"{v:.2f}", (u, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7)
        ax.set_xlabel("Subject ID", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} per subject", fontsize=11, fontweight="bold")
        ax.set_xticks(users)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_metric_distributions(folds):
    maes    = [f["test_mae"]       for f in folds]
    r2s     = [f["test_r2"]        for f in folds]
    mean_rs = [f["mean_pearson_r"] for f in folds]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    data_triples = [
        (maes,    "MAE (°)",       "steelblue"),
        (r2s,     "R²",            "coral"),
        (mean_rs, "Mean Pearson r", "mediumseagreen"),
    ]
    for ax, (vals, label, color) in zip(axes, data_triples):
        vp = ax.violinplot(vals, positions=[1], showmedians=True, showextrema=True)
        for pc in vp["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.6)
        vp["cmedians"].set_color("#333"); vp["cmedians"].set_linewidth(2)
        ax.scatter([1] * len(vals), vals, color=color, s=30, zorder=3,
                   edgecolors="white", linewidths=0.5, alpha=0.9)
        ax.set_title(f"{label}\nμ={np.mean(vals):.3f}  σ={np.std(vals):.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Distribution across 21 subjects (violin + individual points)", fontsize=11)
    fig.tight_layout()
    return fig


def _plot_mae_r2_correlation(folds):
    maes    = [f["test_mae"]       for f in folds]
    r2s     = [f["test_r2"]        for f in folds]
    mean_rs = [f["mean_pearson_r"] for f in folds]
    users   = [f["test_user"]      for f in folds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=min(maes), vmax=max(maes))
    colors = [cmap(norm(m)) for m in maes]

    sc1 = ax1.scatter(maes, r2s, c=colors, s=80, edgecolors="white", linewidths=0.5, zorder=3)
    for u, m, r in zip(users, maes, r2s):
        ax1.annotate(f"U{u:02d}", (m, r), fontsize=7, ha="center",
                     xytext=(0, 7), textcoords="offset points")
    ax1.set_xlabel("MAE (°)", fontsize=10); ax1.set_ylabel("R²", fontsize=10)
    ax1.set_title("MAE vs R² across subjects", fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.3)
    corr1 = np.corrcoef(maes, r2s)[0, 1]
    ax1.text(0.05, 0.05, f"Pearson r = {corr1:.3f}", transform=ax1.transAxes,
             fontsize=9, color="#333")

    sc2 = ax2.scatter(maes, mean_rs, c=colors, s=80, edgecolors="white", linewidths=0.5, zorder=3)
    for u, m, r in zip(users, maes, mean_rs):
        ax2.annotate(f"U{u:02d}", (m, r), fontsize=7, ha="center",
                     xytext=(0, 7), textcoords="offset points")
    ax2.set_xlabel("MAE (°)", fontsize=10); ax2.set_ylabel("Mean Pearson r", fontsize=10)
    ax2.set_title("MAE vs Mean Pearson r across subjects", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.3)
    corr2 = np.corrcoef(maes, mean_rs)[0, 1]
    ax2.text(0.05, 0.05, f"Pearson r = {corr2:.3f}", transform=ax2.transAxes,
             fontsize=9, color="#333")

    fig.tight_layout()
    return fig


# ── section 2: per-finger ─────────────────────────────────────────────────────

def _plot_per_finger_metrics(folds):
    fingers = list(FINGER_GROUPS.keys())
    finger_mae = {f: [] for f in fingers}
    finger_r   = {f: [] for f in fingers}

    for fold in folds:
        for finger, (lo, hi) in FINGER_GROUPS.items():
            joint_maes = [fold["per_joint_mae"][JOINT_NAMES[j]] for j in range(lo, hi)]
            joint_rs   = [fold["per_joint_r"][JOINT_NAMES[j]]   for j in range(lo, hi)]
            finger_mae[finger].append(np.mean(joint_maes))
            finger_r[finger].append(np.mean(joint_rs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(fingers))
    width = 0.35

    means_mae = [np.mean(finger_mae[f]) for f in fingers]
    stds_mae  = [np.std(finger_mae[f])  for f in fingers]
    means_r   = [np.mean(finger_r[f])   for f in fingers]
    stds_r    = [np.std(finger_r[f])    for f in fingers]
    colors    = [FINGER_COLORS[f] for f in fingers]

    bars1 = ax1.bar(x, means_mae, yerr=stds_mae, color=colors,
                    capsize=5, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels([f.capitalize() for f in fingers], fontsize=10)
    ax1.set_ylabel("Mean MAE (°)", fontsize=10)
    ax1.set_title("Per-finger MAE (mean ± std across subjects)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar, m, s in zip(bars1, means_mae, stds_mae):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.1,
                 f"{m:.2f}°", ha="center", fontsize=9)

    bars2 = ax2.bar(x, means_r, yerr=stds_r, color=colors,
                    capsize=5, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels([f.capitalize() for f in fingers], fontsize=10)
    ax2.set_ylabel("Mean Pearson r", fontsize=10)
    ax2.set_title("Per-finger Pearson r (mean ± std across subjects)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar, m, s in zip(bars2, means_r, stds_r):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                 f"{m:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    return fig


def _plot_per_finger_boxplot(folds):
    fingers = list(FINGER_GROUPS.keys())
    finger_mae = {f: [] for f in fingers}

    for fold in folds:
        for finger, (lo, hi) in FINGER_GROUPS.items():
            joint_maes = [fold["per_joint_mae"][JOINT_NAMES[j]] for j in range(lo, hi)]
            finger_mae[finger].append(np.mean(joint_maes))

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [finger_mae[f] for f in fingers]
    colors = [FINGER_COLORS[f] for f in fingers]

    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.RandomState(42).uniform(-0.1, 0.1, len(d))
        ax.scatter([i] * len(d) + jitter, d, color=color, s=25, zorder=3,
                   edgecolors="white", linewidths=0.3)

    ax.set_xticks(range(1, len(fingers) + 1))
    ax.set_xticklabels([f.capitalize() for f in fingers], fontsize=11)
    ax.set_ylabel("Mean Finger MAE (°) per subject", fontsize=10)
    ax.set_title("Per-finger MAE distribution across 21 subjects", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── section 3: per-joint ──────────────────────────────────────────────────────

def _plot_per_joint_mean_bar(folds):
    matrix_mae = np.array([[f["per_joint_mae"][j] for j in JOINT_NAMES] for f in folds])
    matrix_r   = np.array([[f["per_joint_r"][j]   for j in JOINT_NAMES] for f in folds])
    mean_mae   = matrix_mae.mean(axis=0)
    std_mae    = matrix_mae.std(axis=0)
    mean_r     = matrix_r.mean(axis=0)
    std_r      = matrix_r.std(axis=0)

    colors = _joint_colors()
    x = np.arange(20)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))

    bars1 = ax1.bar(x, mean_mae, yerr=std_mae, color=colors, capsize=3,
                    edgecolor="white", linewidth=0.4)
    ax1.axhline(mean_mae.mean(), color="black", linestyle="--", linewidth=1.5,
                label=f"overall mean = {mean_mae.mean():.2f}°")
    ax1.set_xticks(x)
    ax1.set_xticklabels(JOINT_SHORT, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Mean MAE (°) ± std", fontsize=10)
    ax1.set_title("Per-joint MAE averaged over 21 subjects (error bars = std across subjects)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
    legend_patches = [Patch(color=FINGER_COLORS[f], label=f.capitalize()) for f in FINGER_GROUPS]
    ax1.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                   label=f"overall mean = {mean_mae.mean():.2f}°")
    ], fontsize=9, loc="upper right")

    bars2 = ax2.bar(x, mean_r, yerr=std_r, color=colors, capsize=3,
                    edgecolor="white", linewidth=0.4)
    ax2.axhline(mean_r.mean(), color="black", linestyle="--", linewidth=1.5,
                label=f"overall mean r = {mean_r.mean():.3f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(JOINT_SHORT, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean Pearson r ± std", fontsize=10)
    ax2.set_title("Per-joint Pearson r averaged over 21 subjects",
                  fontsize=11, fontweight="bold")
    ax2.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                   label=f"overall mean r = {mean_r.mean():.3f}")
    ], fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_joint_heatmap(folds, metric_key, title, cmap, vmin=None, vmax=None):
    users  = [f["test_user"] for f in folds]
    matrix = np.array([[f[metric_key][j] for j in JOINT_NAMES] for f in folds])

    fig, ax = plt.subplots(figsize=(17, 8))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   vmin=vmin if vmin is not None else matrix.min(),
                   vmax=vmax if vmax is not None else matrix.max())
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(20))
    ax.set_xticklabels(JOINT_SHORT, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(users)))
    ax.set_yticklabels([f"User{u:02d}" for u in users], fontsize=8.5)
    ax.set_title(title, fontsize=12, fontweight="bold")

    mid = matrix.mean()
    for i in range(len(users)):
        for j in range(20):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if v > mid else "black")

    # draw finger group separators
    for _, (lo, _) in list(FINGER_GROUPS.items())[1:]:
        ax.axvline(lo - 0.5, color="white", linewidth=1.5, alpha=0.7)

    fig.tight_layout()
    return fig


def _plot_best_worst_comparison(folds):
    maes       = [f["test_mae"] for f in folds]
    best_fold  = folds[int(np.argmin(maes))]
    worst_fold = folds[int(np.argmax(maes))]

    best_mae  = [best_fold["per_joint_mae"][j]  for j in JOINT_NAMES]
    worst_mae = [worst_fold["per_joint_mae"][j] for j in JOINT_NAMES]
    best_r    = [best_fold["per_joint_r"][j]    for j in JOINT_NAMES]
    worst_r   = [worst_fold["per_joint_r"][j]   for j in JOINT_NAMES]

    x = np.arange(20)
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))

    ax1.bar(x - width / 2, best_mae,  width, label=f"User{best_fold['test_user']:02d} (best,  {best_fold['test_mae']:.2f}°)",
            color="#34d399", edgecolor="white", linewidth=0.4)
    ax1.bar(x + width / 2, worst_mae, width, label=f"User{worst_fold['test_user']:02d} (worst, {worst_fold['test_mae']:.2f}°)",
            color="#f87171", edgecolor="white", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(JOINT_SHORT, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("MAE (°)", fontsize=10)
    ax1.set_title("Per-joint MAE: best vs worst subject", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x - width / 2, best_r,  width, label=f"User{best_fold['test_user']:02d} (best)",
            color="#34d399", edgecolor="white", linewidth=0.4)
    ax2.bar(x + width / 2, worst_r, width, label=f"User{worst_fold['test_user']:02d} (worst)",
            color="#f87171", edgecolor="white", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(JOINT_SHORT, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Pearson r", fontsize=10)
    ax2.set_title("Per-joint Pearson r: best vs worst subject", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ── section 4: training curves ────────────────────────────────────────────────

def _plot_training_curves_grid(exp_dir, folds):
    nrows, ncols = 4, 6  # 24 cells, 21 used
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 13))
    axes_flat = axes.flatten()

    for i, fold in enumerate(folds):
        ax = axes_flat[i]
        fold_dir = exp_dir / f"fold{fold['fold']:02d}_testUser{fold['test_user']:02d}"
        history = _load_history(fold_dir)

        if not history:
            ax.text(0.5, 0.5, "no history", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        else:
            epochs   = [h["epoch"]    for h in history]
            val_maes = [h["val_mae"]  for h in history]
            tr_loss  = [h["train_loss"] for h in history]
            best_ep  = epochs[int(np.argmin(val_maes))]
            ax2 = ax.twinx()
            ax.plot(epochs, tr_loss,  color="steelblue", lw=1.2, alpha=0.85, label="train loss")
            ax2.plot(epochs, val_maes, color="coral",    lw=1.2, alpha=0.85, label="val MAE")
            ax.axvline(best_ep, color="#333", linestyle=":", lw=0.8)
            ax2.tick_params(labelsize=5, colors="coral")
            ax.tick_params(labelsize=5, colors="steelblue")
            ax.yaxis.label.set_color("steelblue")
            ax2.yaxis.label.set_color("coral")

        color = plt.cm.RdYlGn_r(mcolors.Normalize(3.5, 10.5)(fold["test_mae"]))
        ax.set_title(f"U{fold['test_user']:02d}  {fold['test_mae']:.2f}°",
                     fontsize=8, color="black",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.4))
        ax.grid(alpha=0.2)

    # hide spare axes
    for j in range(len(folds), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color="steelblue", lw=2, label="train loss (SmoothL1)"),
        plt.Line2D([0], [0], color="coral",     lw=2, label="val MAE (normalised)"),
        plt.Line2D([0], [0], color="#333", linestyle=":", lw=1.5, label="best epoch"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.99, 0.01))
    fig.suptitle("Training curves per fold — title shows test subject and test MAE (°)",
                 fontsize=13, fontweight="bold", y=1.002)
    fig.tight_layout(h_pad=1.2, w_pad=0.5)
    return fig


def _plot_training_convergence(exp_dir, folds):
    n_epochs = []
    best_val = []
    for fold in folds:
        fold_dir = exp_dir / f"fold{fold['fold']:02d}_testUser{fold['test_user']:02d}"
        history  = _load_history(fold_dir)
        if history:
            n_epochs.append(len(history))
            best_val.append(min(h["val_mae"] for h in history))
        else:
            n_epochs.append(0); best_val.append(float("nan"))

    maes  = [f["test_mae"] for f in folds]
    users = [f["test_user"] for f in folds]
    norm  = mcolors.Normalize(vmin=min(maes), vmax=max(maes))
    colors = [plt.cm.RdYlGn_r(norm(m)) for m in maes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar([f"U{u:02d}" for u in users], n_epochs, color=colors,
            edgecolor="white", linewidth=0.4)
    ax1.set_xlabel("Subject"); ax1.set_ylabel("Epochs until early stop")
    ax1.set_title("Training epochs per fold", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45, labelsize=7)
    ax1.grid(axis="y", alpha=0.3)

    ax2.scatter(n_epochs, maes, c=colors, s=70, edgecolors="white", linewidths=0.5, zorder=3)
    for u, ep, m in zip(users, n_epochs, maes):
        ax2.annotate(f"U{u:02d}", (ep, m), fontsize=7, ha="center",
                     xytext=(0, 7), textcoords="offset points")
    ax2.set_xlabel("Training epochs"); ax2.set_ylabel("Test MAE (°)")
    ax2.set_title("Epochs vs Test MAE", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ── section 5: joint difficulty ranking ──────────────────────────────────────

def _joint_ranking_md(folds):
    matrix_mae = np.array([[f["per_joint_mae"][j] for j in JOINT_NAMES] for f in folds])
    matrix_r   = np.array([[f["per_joint_r"][j]   for j in JOINT_NAMES] for f in folds])
    mean_mae   = matrix_mae.mean(axis=0)
    std_mae    = matrix_mae.std(axis=0)
    mean_r     = matrix_r.mean(axis=0)

    order = np.argsort(mean_mae)[::-1]
    rows = []
    for rank, idx in enumerate(order, start=1):
        finger = _finger_for_joint(idx).capitalize()
        rows.append(
            f"| {rank} | {JOINT_NAMES[idx]} | {finger} | "
            f"**{mean_mae[idx]:.2f}°** | {std_mae[idx]:.2f}° | {mean_r[idx]:.3f} |"
        )
    header = (
        "## Joint Difficulty Ranking (hardest → easiest)\n\n"
        "| Rank | Joint | Finger | Mean MAE | Std MAE | Mean Pearson r |\n"
        "|------|-------|--------|----------|---------|----------------|\n"
    )
    return header + "\n".join(rows)


# ── notebook assembly ─────────────────────────────────────────────────────────

def build_report(exp_dir):
    summary_path = exp_dir / "loso_summary.json"
    summary = json.loads(summary_path.read_text())
    folds   = summary["per_fold"]
    run     = summary["run"]

    maes    = [f["test_mae"]       for f in folds]
    r2s     = [f["test_r2"]        for f in folds]
    mean_rs = [f["mean_pearson_r"] for f in folds]
    rmses   = [f["test_rmse"]      for f in folds]
    best_fold  = folds[int(np.argmin(maes))]
    worst_fold = folds[int(np.argmax(maes))]

    cells = []

    # ── 0. Title ─────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source=(
        f"# LOSO-CV Detailed Report\n"
        f"## {run}\n\n"
        f"**Model**: PoseNetSeq (TCN stem + 2-layer BiGRU)  \n"
        f"**Features**: 16-channel EMG (bandpass-filtered + RMS envelope, per-user z-score)  \n"
        f"**Labels**: 20 hand joint angles (5 fingers × 4 DOF), per-user z-score → denormalized to °  \n"
        f"**Dataset**: PiMForce — 21 subjects, 3 sessions each  \n"
        f"**Protocol**: Leave-One-Subject-Out CV, 21 folds  \n"
        f"**Config**: lr=3e-4 · batch=64 · max_epochs=60 · patience=20 · wd=1e-2 · seq_len=64"
    )))

    # ── 1. Overall metrics ────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source=(
        "## 1 · Overall Metrics (21-fold LOSO)\n\n"
        "| Metric | Mean | Std | Min | Max |\n"
        "|--------|------|-----|-----|-----|\n"
        f"| MAE (°) | **{np.mean(maes):.4f}** | {np.std(maes):.4f} | {min(maes):.4f} | {max(maes):.4f} |\n"
        f"| RMSE (°) | {np.mean(rmses):.4f} | {np.std(rmses):.4f} | {min(rmses):.4f} | {max(rmses):.4f} |\n"
        f"| R² | {np.mean(r2s):.4f} | {np.std(r2s):.4f} | {min(r2s):.4f} | {max(r2s):.4f} |\n"
        f"| Mean Pearson r | {np.mean(mean_rs):.4f} | {np.std(mean_rs):.4f} | {min(mean_rs):.4f} | {max(mean_rs):.4f} |\n\n"
        f"**Best subject**: User{best_fold['test_user']:02d} — MAE={best_fold['test_mae']:.4f}°, R²={best_fold['test_r2']:.4f}  \n"
        f"**Worst subject**: User{worst_fold['test_user']:02d} — MAE={worst_fold['test_mae']:.4f}°, R²={worst_fold['test_r2']:.4f}  \n"
        f"**Inter-subject range**: {max(maes) - min(maes):.2f}° ({max(maes)/min(maes):.1f}× worst/best ratio)"
    )))

    # ── 2. Per-subject table ──────────────────────────────────────────────────
    rows_sorted = sorted(folds, key=lambda x: x["test_mae"])
    rows_md = "\n".join(
        f"| {'🟢' if f['test_user']==best_fold['test_user'] else '🔴' if f['test_user']==worst_fold['test_user'] else ''}"
        f"User{f['test_user']:02d} | {f['test_mae']:.4f} | {f['test_rmse']:.4f} | "
        f"{f['test_r2']:.4f} | {f['mean_pearson_r']:.4f} |"
        for f in rows_sorted
    )
    cells.append(new_markdown_cell(source=(
        "## 2 · Per-subject Results (sorted by MAE)\n\n"
        "| Subject | MAE (°) | RMSE (°) | R² | Mean Pearson r |\n"
        "|---------|---------|----------|----|----------------|\n"
        + rows_md
    )))

    # ── 3. Per-subject bar ────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## 3 · Per-subject MAE Bar Chart"))
    fig = _plot_per_subject_bar(folds, np.mean(maes))
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 4. Per-subject scatter ────────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## 4 · Per-subject Metrics (MAE, R², Pearson r)"))
    fig = _plot_three_metrics_scatter(folds)
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 5. Distributions ─────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## 5 · Metric Distributions Across Subjects"))
    fig = _plot_metric_distributions(folds)
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 6. MAE vs R² correlation ──────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## 6 · Cross-subject Metric Correlations"))
    fig = _plot_mae_r2_correlation(folds)
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 7. Per-finger ─────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## 7 · Per-finger Metrics"))
    fig = _plot_per_finger_metrics(folds)
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    cells.append(new_markdown_cell(source="### Per-finger MAE Distribution (box plot)"))
    fig = _plot_per_finger_boxplot(folds)
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 8. Per-joint bars ─────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## 8 · Per-joint Metrics (averaged over all subjects)"))
    fig = _plot_per_joint_mean_bar(folds)
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 9. Joint difficulty ranking ───────────────────────────────────────────
    cells.append(new_markdown_cell(source=_joint_ranking_md(folds)))

    # ── 10. Per-joint MAE heatmap ─────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## 9 · Per-joint MAE Heatmap (subject × joint)"))
    fig = _plot_joint_heatmap(folds, "per_joint_mae",
                              "Per-joint MAE (°) — rows = subjects, cols = joints, vertical lines = finger boundaries",
                              "YlOrRd")
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 11. Per-joint Pearson r heatmap ──────────────────────────────────────
    cells.append(new_markdown_cell(source="## 10 · Per-joint Pearson r Heatmap (subject × joint)"))
    fig = _plot_joint_heatmap(folds, "per_joint_r",
                              "Per-joint Pearson r — rows = subjects, cols = joints",
                              "RdYlGn", vmin=0.0, vmax=1.0)
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 12. Best vs worst comparison ─────────────────────────────────────────
    cells.append(new_markdown_cell(source="## 11 · Best vs Worst Subject — Per-joint Comparison"))
    fig = _plot_best_worst_comparison(folds)
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 13. Training curves grid ──────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## 12 · Training Curves — All Folds"))
    fig = _plot_training_curves_grid(exp_dir, folds)
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 14. Convergence ───────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## 13 · Convergence Summary"))
    fig = _plot_training_convergence(exp_dir, folds)
    cells.append(_img_cell(_fig_to_b64(fig))); plt.close(fig)

    # ── 15. Key findings ──────────────────────────────────────────────────────
    matrix_mae = np.array([[f["per_joint_mae"][j] for j in JOINT_NAMES] for f in folds])
    mean_joint_mae = matrix_mae.mean(axis=0)
    hardest_joint  = JOINT_NAMES[int(np.argmax(mean_joint_mae))]
    easiest_joint  = JOINT_NAMES[int(np.argmin(mean_joint_mae))]

    cells.append(new_markdown_cell(source=(
        "## 14 · Key Findings\n\n"
        f"- **Overall MAE**: {np.mean(maes):.2f}° ± {np.std(maes):.2f}° across 21 subjects\n"
        f"- **R²**: {np.mean(r2s):.3f} ± {np.std(r2s):.3f} — model explains {np.mean(r2s)*100:.1f}% of variance on average\n"
        f"- **Best subject**: User{best_fold['test_user']:02d} ({best_fold['test_mae']:.2f}° MAE, R²={best_fold['test_r2']:.3f})\n"
        f"- **Worst subject**: User{worst_fold['test_user']:02d} ({worst_fold['test_mae']:.2f}° MAE, R²={worst_fold['test_r2']:.3f})\n"
        f"- **Worst/best ratio**: {max(maes)/min(maes):.1f}× — large inter-subject variability\n"
        f"- **Hardest joint**: `{hardest_joint}` ({mean_joint_mae.max():.2f}°)\n"
        f"- **Easiest joint**: `{easiest_joint}` ({mean_joint_mae.min():.2f}°)\n"
        f"- **MCP-stretch joints** (index, middle, ring, pinky) consistently show highest MAE — these are the primary flexion DOFs with high range of motion\n"
        f"- **MCP-spread joints** show lowest MAE — low range of motion makes them easier to regress\n"
        f"- Training converged in 21–41 epochs (early stopping from 60 max), suggesting the model is not underfitting"
    )))

    # ── save ──────────────────────────────────────────────────────────────────
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    out_path = exp_dir / "loso_report_detailed.ipynb"
    out_path.write_text(nbformat.writes(nb))
    print(f"\nSaved: {out_path}")
    return out_path


def _find_latest_loso(experiments_dir):
    runs = sorted(
        [p for p in experiments_dir.iterdir()
         if p.is_dir() and "_loso" in p.name],
        key=lambda p: p.name,
    )
    if not runs:
        raise FileNotFoundError("No LOSO experiment directories found.")
    return runs[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", nargs="?", default=None)
    args = parser.parse_args()

    experiments_dir = ROOT / "experiments"
    exp_dir = Path(args.exp_dir) if args.exp_dir else _find_latest_loso(experiments_dir)
    print(f"Building detailed report for: {exp_dir.name}")
    build_report(exp_dir)


if __name__ == "__main__":
    main()
