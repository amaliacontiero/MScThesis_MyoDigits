"""
Generate a LOSO CV summary notebook from a completed experiment.

Usage (from project root):
    python scripts/loso_report.py                         # latest loso run
    python scripts/loso_report.py experiments/run02_...  # specific run
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
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[2]

JOINT_NAMES = [
    "thumb-cmc-spread",  "thumb-cmc-stretch",  "thumb-mcp-stretch", "thumb-ip-stretch",
    "index-mcp-spread",  "index-mcp-stretch",  "index-pip-stretch", "index-dip-stretch",
    "middle-mcp-spread", "middle-mcp-stretch", "middle-pip-stretch","middle-dip-stretch",
    "ring-mcp-spread",   "ring-mcp-stretch",   "ring-pip-stretch",  "ring-dip-stretch",
    "pinky-mcp-spread",  "pinky-mcp-stretch",  "pinky-pip-stretch", "pinky-dip-stretch",
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


def _find_latest_loso(experiments_dir):
    runs = sorted(
        [p for p in experiments_dir.iterdir()
         if p.is_dir() and "_loso" in p.name],
        key=lambda p: p.name,
    )
    if not runs:
        raise FileNotFoundError("No LOSO experiment directories found.")
    return runs[-1]


# ── plots ─────────────────────────────────────────────────────────────────────

def _plot_per_fold_bar(folds, mean_mae):
    users  = [f["test_user"] for f in folds]
    maes   = [f["test_mae"]  for f in folds]
    order  = np.argsort(maes)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap   = plt.cm.RdYlGn_r
    norm   = mcolors.Normalize(vmin=min(maes), vmax=max(maes))
    colors = [cmap(norm(m)) for m in [maes[i] for i in order]]

    bars = ax.barh(
        [f"User{users[i]:02d}" for i in order],
        [maes[i] for i in order],
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.axvline(mean_mae, color="black", linestyle="--", linewidth=1.5,
               label=f"mean = {mean_mae:.2f}°")

    for bar, i in zip(bars, order):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{maes[i]:.2f}°", va="center", fontsize=8)

    ax.set_xlabel("Test MAE (°)")
    ax.set_title("Per-subject MAE (sorted, green=best, red=worst)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_metrics_scatter(folds):
    users   = [f["test_user"]      for f in folds]
    maes    = [f["test_mae"]       for f in folds]
    r2s     = [f["test_r2"]        for f in folds]
    mean_rs = [f["mean_pearson_r"] for f in folds]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, vals, label, color in zip(
        axes,
        [maes, r2s, mean_rs],
        ["MAE (°)", "R²", "Mean Pearson r"],
        ["steelblue", "coral", "mediumseagreen"],
    ):
        ax.scatter(users, vals, color=color, s=60, zorder=3)
        ax.axhline(np.mean(vals), color="black", linestyle="--", linewidth=1,
                   label=f"mean={np.mean(vals):.3f}")
        for u, v in zip(users, vals):
            ax.annotate(f"{v:.2f}", (u, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7)
        ax.set_xlabel("Subject")
        ax.set_ylabel(label)
        ax.set_title(label + " per subject")
        ax.set_xticks(users)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_joint_heatmap(folds, metric_key, title, cmap):
    users  = [f["test_user"] for f in folds]
    matrix = np.array([[f[metric_key][j] for j in JOINT_NAMES] for f in folds])

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(20))
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(users)))
    ax.set_yticklabels([f"User{u:02d}" for u in users], fontsize=9)
    ax.set_title(title)

    for i in range(len(users)):
        for j in range(20):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=5.5, color="white" if matrix[i, j] > matrix.mean() else "black")

    fig.tight_layout()
    return fig


def _plot_mean_joint_bar(folds):
    matrix_mae = np.array([[f["per_joint_mae"][j] for j in JOINT_NAMES] for f in folds])
    matrix_r   = np.array([[f["per_joint_r"][j]   for j in JOINT_NAMES] for f in folds])
    mean_mae   = matrix_mae.mean(axis=0)
    mean_r     = matrix_r.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.bar(range(20), mean_mae, color="steelblue")
    ax1.axhline(mean_mae.mean(), color="red", linestyle="--",
                label=f"overall mean={mean_mae.mean():.2f}°")
    ax1.set_xticks(range(20))
    ax1.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Mean MAE across folds (°)")
    ax1.set_title("Per-joint MAE averaged over all subjects")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    ax2.bar(range(20), mean_r, color="coral")
    ax2.axhline(mean_r.mean(), color="red", linestyle="--",
                label=f"overall mean r={mean_r.mean():.3f}")
    ax2.set_xticks(range(20))
    ax2.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean Pearson r across folds")
    ax2.set_title("Per-joint Pearson r averaged over all subjects")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


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

    # ── header ────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source=f"# LOSO CV Report — {run}"))

    # ── overall metrics table ─────────────────────────────────────────────────
    cells.append(new_markdown_cell(source=(
        "## Overall Metrics\n\n"
        f"| Metric | Mean | Std |\n"
        f"|--------|------|-----|\n"
        f"| MAE (°) | **{summary['mean_test_mae']:.4f}** | {summary['std_test_mae']:.4f} |\n"
        f"| RMSE (°) | {summary['mean_test_rmse']:.4f} | {summary['std_test_rmse']:.4f} |\n"
        f"| R² | {summary['mean_test_r2']:.4f} | {summary['std_test_r2']:.4f} |\n"
        f"| Mean Pearson r | {summary['mean_pearson_r']:.4f} | {summary['std_pearson_r']:.4f} |\n"
        f"| Folds | {summary['n_folds']} | — |\n"
    )))

    # ── best / worst ──────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source=(
        "## Best & Worst Subjects\n\n"
        f"| | Subject | MAE (°) | RMSE (°) | R² | Mean r |\n"
        f"|---|---------|---------|----------|----|--------|\n"
        f"| 🟢 Best  | User{best_fold['test_user']:02d} | "
        f"**{best_fold['test_mae']:.4f}** | {best_fold['test_rmse']:.4f} | "
        f"{best_fold['test_r2']:.4f} | {best_fold['mean_pearson_r']:.4f} |\n"
        f"| 🔴 Worst | User{worst_fold['test_user']:02d} | "
        f"**{worst_fold['test_mae']:.4f}** | {worst_fold['test_rmse']:.4f} | "
        f"{worst_fold['test_r2']:.4f} | {worst_fold['mean_pearson_r']:.4f} |\n"
    )))

    # ── per-fold table ────────────────────────────────────────────────────────
    rows = "\n".join(
        f"| User{f['test_user']:02d} | {f['test_mae']:.4f} | "
        f"{f['test_rmse']:.4f} | {f['test_r2']:.4f} | {f['mean_pearson_r']:.4f} |"
        for f in sorted(folds, key=lambda x: x["test_mae"])
    )
    cells.append(new_markdown_cell(source=(
        "## Per-subject Results (sorted by MAE)\n\n"
        "| Subject | MAE (°) | RMSE (°) | R² | Mean Pearson r |\n"
        "|---------|---------|----------|----|----------------|\n"
        + rows
    )))

    # ── per-fold bar chart ────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## Per-subject MAE Bar Chart"))
    fig = _plot_per_fold_bar(folds, summary["mean_test_mae"])
    cells.append(_img_cell(_fig_to_b64(fig), "per-subject MAE bar chart"))
    plt.close(fig)

    # ── metrics scatter ───────────────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## Per-subject Metrics Overview"))
    fig = _plot_metrics_scatter(folds)
    cells.append(_img_cell(_fig_to_b64(fig), "per-subject metrics scatter"))
    plt.close(fig)

    # ── mean per-joint bars ───────────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## Per-joint Metrics (averaged across subjects)"))
    fig = _plot_mean_joint_bar(folds)
    cells.append(_img_cell(_fig_to_b64(fig), "mean per-joint metrics"))
    plt.close(fig)

    # ── per-joint MAE heatmap ─────────────────────────────────────────────────
    cells.append(new_markdown_cell(source="## Per-joint MAE Heatmap (subject × joint)"))
    fig = _plot_joint_heatmap(folds, "per_joint_mae", "Per-joint MAE (°) — each row is one subject", "YlOrRd")
    cells.append(_img_cell(_fig_to_b64(fig), "per-joint MAE heatmap"))
    plt.close(fig)

    # ── per-joint Pearson r heatmap ───────────────────────────────────────────
    cells.append(new_markdown_cell(source="## Per-joint Pearson r Heatmap (subject × joint)"))
    fig = _plot_joint_heatmap(folds, "per_joint_r", "Per-joint Pearson r — each row is one subject", "RdYlGn")
    cells.append(_img_cell(_fig_to_b64(fig), "per-joint Pearson r heatmap"))
    plt.close(fig)

    # ── save ──────────────────────────────────────────────────────────────────
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    out_path = exp_dir / "loso_report.ipynb"
    out_path.write_text(nbformat.writes(nb))
    print(f"Saved: {out_path}")
    return out_path


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", nargs="?", default=None,
                        help="Path to LOSO experiment directory (default: latest)")
    args = parser.parse_args()

    experiments_dir = ROOT / "experiments"
    exp_dir = Path(args.exp_dir) if args.exp_dir else _find_latest_loso(experiments_dir)
    print(f"Generating report for: {exp_dir.name}")
    build_report(exp_dir)


if __name__ == "__main__":
    main()
