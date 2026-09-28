"""
Post-training evaluation.

Produces inside exp_dir:
  metrics.json          - MAE, RMSE, R², per-joint MAE, per-joint Pearson r
  plots/training.png    - train loss + val MAE curves
  plots/per_joint.png   - per-joint MAE and Pearson r bars
  hand_animation.mp4    - 3-D hand skeleton GT vs prediction
  results.ipynb         - notebook with all figures embedded (open, no re-run needed)
"""

import io
import base64
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from scipy.stats import pearsonr
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D

JOINT_NAMES = [
    "thumb-cmc-spread",  "thumb-cmc-stretch",  "thumb-mcp-stretch", "thumb-ip-stretch",
    "index-mcp-spread",  "index-mcp-stretch",  "index-pip-stretch", "index-dip-stretch",
    "middle-mcp-spread", "middle-mcp-stretch", "middle-pip-stretch","middle-dip-stretch",
    "ring-mcp-spread",   "ring-mcp-stretch",   "ring-pip-stretch",  "ring-dip-stretch",
    "pinky-mcp-spread",  "pinky-mcp-stretch",  "pinky-pip-stretch", "pinky-dip-stretch",
]

# ── hand forward kinematics (same as 5_eval.ipynb) ───────────────────────────
BONE_LEN = {
    "thumb":  (40, 32, 25),
    "index":  (45, 28, 20),
    "middle": (50, 32, 22),
    "ring":   (45, 28, 20),
    "pinky":  (32, 20, 16),
}
MCP = {
    "index":  np.array([-30.,  90., 0.]),
    "middle": np.array([-10.,  95., 0.]),
    "ring":   np.array([ 10.,  90., 0.]),
    "pinky":  np.array([ 28.,  80., 0.]),
}
CMC  = np.array([-42., 52., 0.])
PALM = np.array([
    [-42.,  2., 0.], [ 35.,  2., 0.], [ 32.,  6., 0.],
    MCP["pinky"], MCP["ring"], MCP["middle"], MCP["index"], CMC,
])
FINGER_COLOR = {
    "thumb": "#a78bfa", "index": "#60a5fa",
    "middle": "#34d399", "ring": "#f87171", "pinky": "#fb923c",
}


def _Rx(d):
    r = np.radians(d); c, s = np.cos(r), np.sin(r)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def _Rz(d):
    r = np.radians(d); c, s = np.cos(r), np.sin(r)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def _Ry(d):
    r = np.radians(d); c, s = np.cos(r), np.sin(r)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def _T(R, t):
    M = np.eye(4); M[:3,:3] = R; M[:3,3] = t; return M

def _pt(M):
    return M[:3, 3].copy()

def _finger_fk(origin, spread, mcp_s, pip_s, dip_s, lengths):
    l0, l1, l2 = lengths
    M = _T(_Rz(spread) @ _Rx(-mcp_s), origin)
    mcp = _pt(M)
    M = M @ _T(_Rx(-pip_s), [0, l0, 0]); pip = _pt(M)
    M = M @ _T(_Rx(-dip_s), [0, l1, 0]); dip = _pt(M)
    M = M @ _T(np.eye(3),   [0, l2, 0]); tip = _pt(M)
    return [mcp, pip, dip, tip]

def _thumb_fk(cmc_origin, cmc_spr, cmc_s, mcp_s, ip_s, lengths):
    l0, l1, l2 = lengths
    M = _T(_Rz(-55) @ _Ry(12), cmc_origin)
    M = M @ _T(_Rz(cmc_spr) @ _Rx(-cmc_s), [0, 0, 0]); cmc = _pt(M)
    M = M @ _T(_Rx(-mcp_s), [0, l0, 0]);                 mcp = _pt(M)
    M = M @ _T(_Rx(-ip_s),  [0, l1, 0]);                 ip  = _pt(M)
    M = M @ _T(np.eye(3),   [0, l2, 0]);                 tip = _pt(M)
    return [cmc, mcp, ip, tip]

def _hand_fk(a):
    return {
        "thumb":  _thumb_fk (CMC,           a[0], a[1], a[2],  a[3],  BONE_LEN["thumb"] ),
        "index":  _finger_fk(MCP["index"],  a[4], a[5], a[6],  a[7],  BONE_LEN["index"] ),
        "middle": _finger_fk(MCP["middle"], a[8], a[9], a[10], a[11], BONE_LEN["middle"]),
        "ring":   _finger_fk(MCP["ring"],   a[12],a[13],a[14], a[15], BONE_LEN["ring"]  ),
        "pinky":  _finger_fk(MCP["pinky"],  a[16],a[17],a[18], a[19], BONE_LEN["pinky"] ),
    }


# ── notebook helpers ──────────────────────────────────────────────────────────
def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _image_output_cell(b64, title=""):
    cell = new_code_cell(source=f"# {title}" if title else "# figure")
    cell.outputs = [nbformat.v4.new_output(
        output_type="display_data",
        data={"image/png": b64, "text/plain": ["<Figure>"]},
        metadata={},
    )]
    cell.metadata["tags"] = ["hide-input"]
    return cell


# ── main entry point ──────────────────────────────────────────────────────────
def run_eval(model, loader_test, history, config, device, exp_dir, run_name,
             jnt_denorm=None):
    """
    Parameters
    ----------
    model       : nn.Module - will have best checkpoint loaded automatically
    loader_test : DataLoader
    history     : list of dicts from train()
    config      : dict
    device      : torch.device
    exp_dir     : Path - experiment root (results saved here)
    run_name    : str  - used in titles
    jnt_denorm  : tuple (jnt_mean, jnt_std) as (20,) np.ndarray, or None.
                  When provided, predictions and targets are denormalized to
                  physical units (degrees) before metrics and animation.
    """
    exp_dir   = Path(exp_dir)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(config["checkpoint_dir"])
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, val MAE {ckpt['best_val_mae']:.4f})")
    else:
        print("Warning: best.pt not found, using current model weights")

    # ── inference ────────────────────────────────────────────────────────────
    model.eval()
    use_amp = device.type == "cuda"
    preds_list, targets_list = [], []
    with torch.no_grad():
        for Xb, yb in loader_test:
            Xb = Xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(Xb)
            preds_list.append(out.cpu().float().numpy())
            targets_list.append(yb.numpy())

    preds   = np.concatenate(preds_list,   axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # seq model output: (N, T, 20) -> (N*T, 20)
    if preds.ndim == 3:
        preds   = preds.reshape(-1, 20)
        targets = targets.reshape(-1, 20)

    if jnt_denorm is not None:
        jnt_mean, jnt_std = jnt_denorm
        preds   = preds   * jnt_std + jnt_mean
        targets = targets * jnt_std + jnt_mean
        print("Denormalized predictions and targets to degrees.")

    print(f"Test set: {len(preds):,} windows")

    # ── metrics ──────────────────────────────────────────────────────────────
    err     = preds - targets
    abs_err = np.abs(err)

    mae            = float(abs_err.mean())
    rmse           = float(np.sqrt((err ** 2).mean()))
    ss_res         = (err ** 2).sum()
    # Each joint is measured against its own mean. Pooling all twenty joints
    # against one scalar mean would put the differences BETWEEN joints in the
    # denominator (a flexion angle sits near 50 deg, an abduction angle near 0),
    # which the model gets for free: about 55% of that pooled denominator, so a
    # fixed posture that ignores the sEMG entirely would score roughly 0.55.
    ss_tot         = ((targets - targets.mean(axis=0, keepdims=True)) ** 2).sum()
    r2             = float(1 - ss_res / ss_tot)
    ss_tot_pooled  = ((targets - targets.mean()) ** 2).sum()
    r2_pooled      = float(1 - ss_res / ss_tot_pooled)
    ss_res_j       = (err ** 2).sum(axis=0)
    ss_tot_j       = ((targets - targets.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    per_joint_r2   = (1 - ss_res_j / ss_tot_j).tolist()
    per_joint_mae  = abs_err.mean(axis=0).tolist()
    per_joint_r    = [float(pearsonr(preds[:, j], targets[:, j])[0]) for j in range(20)]
    mean_r         = float(np.mean(per_joint_r))

    metrics = {
        "run": run_name,
        "test_mae":        mae,
        "test_rmse":       rmse,
        "test_r2":         r2,
        "test_r2_pooled":  r2_pooled,
        "mean_pearson_r":  mean_r,
        "per_joint_mae":   {JOINT_NAMES[j]: per_joint_mae[j] for j in range(20)},
        "per_joint_r":     {JOINT_NAMES[j]: per_joint_r[j]   for j in range(20)},
        "per_joint_r2":    {JOINT_NAMES[j]: per_joint_r2[j]  for j in range(20)},
        "n_test_windows":  len(preds),
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  mean_r={mean_r:.4f}")

    # ── plot 1: training curves ───────────────────────────────────────────────
    epochs_     = [r["epoch"]      for r in history]
    train_loss_ = [r["train_loss"] for r in history]
    val_mae_    = [r["val_mae"]    for r in history]

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_, train_loss_, lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss (SmoothL1)")
    ax1.set_title(f"{run_name} -Training loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_, val_mae_, lw=2, color="tab:orange")
    best_epoch_idx = int(np.argmin(val_mae_))
    ax2.axvline(epochs_[best_epoch_idx], color="red", linestyle="--", alpha=0.7,
                label=f"best epoch {epochs_[best_epoch_idx]}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val MAE")
    ax2.set_title(f"{run_name} -Validation MAE")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(plots_dir / "training.png", dpi=150, bbox_inches="tight")
    b64_training = _fig_to_b64(fig1)
    plt.close(fig1)

    # ── plot 2: per-joint ─────────────────────────────────────────────────────
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8))
    x = range(20)

    ax3.bar(x, per_joint_mae, color="steelblue")
    ax3.axhline(mae, color="red", linestyle="--", label=f"mean MAE={mae:.3f}")
    ax3.set_xticks(x); ax3.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("MAE"); ax3.set_title(f"{run_name} -Per-joint MAE"); ax3.legend()
    ax3.grid(True, axis="y", alpha=0.3)

    ax4.bar(x, per_joint_r, color="coral")
    ax4.axhline(mean_r, color="red", linestyle="--", label=f"mean r={mean_r:.3f}")
    ax4.set_xticks(x); ax4.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Pearson r"); ax4.set_title(f"{run_name} -Per-joint Pearson r"); ax4.legend()
    ax4.grid(True, axis="y", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(plots_dir / "per_joint.png", dpi=150, bbox_inches="tight")
    b64_per_joint = _fig_to_b64(fig2)
    plt.close(fig2)

    # ── hand animation ────────────────────────────────────────────────────────
    anim_path = exp_dir / "hand_animation.mp4"
    _save_hand_animation(gt_frames=targets, pred_frames=preds,
                         mae=mae, run_name=run_name, out_path=anim_path)

    # ── results notebook ──────────────────────────────────────────────────────
    _save_notebook(
        exp_dir=exp_dir, run_name=run_name, metrics=metrics,
        b64_training=b64_training, b64_per_joint=b64_per_joint,
        anim_path=anim_path,
    )

    print(f"\nEval complete. Results in {exp_dir}/")
    return metrics


# ── hand animation ────────────────────────────────────────────────────────────
def _init_hand(ax, joints):
    artists = []
    for finger, pts in joints.items():
        fc = FINGER_COLOR[finger]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        line,     = ax.plot(xs, ys, zs, color=fc, lw=2.8, solid_capstyle="round", alpha=0.95)
        scat       = ax.scatter(xs, ys, zs, color=fc, s=25, alpha=0.95,
                                edgecolors="white", linewidths=0.4)
        tip_scat   = ax.scatter([xs[-1]], [ys[-1]], [zs[-1]],
                                color="white", s=45, alpha=0.9)
        artists.append((line, scat, tip_scat))
    return artists

def _update_hand(artists, joints):
    for (line, scat, tip_scat), (_, pts) in zip(artists, joints.items()):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        line.set_data_3d(xs, ys, zs)
        scat._offsets3d     = (np.array(xs),       np.array(ys),       np.array(zs))
        tip_scat._offsets3d = (np.array([xs[-1]]), np.array([ys[-1]]), np.array([zs[-1]]))

def _save_hand_animation(gt_frames, pred_frames, mae, run_name, out_path, n_frames=300):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    search = min(len(gt_frames) - n_frames, 60_000)
    step   = 100
    scores = [gt_frames[i:i+n_frames].var() for i in range(0, search, step)]
    start  = int(np.argmax(scores)) * step
    print(f"Hand animation segment: frames {start}-{start+n_frames}")

    gt_seg   = gt_frames[start:start + n_frames]
    pred_seg = pred_frames[start:start + n_frames]
    mae_per_frame = np.abs(gt_seg - pred_seg).mean(axis=1)

    BG = "#0f0f1a"; PANEL = "#151525"
    fig = plt.figure(figsize=(15, 7.5), facecolor=BG)
    ax_gt = fig.add_subplot(121, projection="3d", facecolor=PANEL)
    ax_pr = fig.add_subplot(122, projection="3d", facecolor=PANEL)

    for ax, label in [(ax_gt, "Ground Truth"), (ax_pr, "Prediction")]:
        ax.set_title(label, color="white", fontsize=13, pad=6, fontweight="bold")
        ax.set_xlim(-80, 80); ax.set_ylim(-20, 175); ax.set_zlim(-80, 60)
        ax.set_xlabel("X", color="#555577", fontsize=7, labelpad=2)
        ax.set_ylabel("Y", color="#555577", fontsize=7, labelpad=2)
        ax.set_zlabel("Z", color="#555577", fontsize=7, labelpad=2)
        ax.tick_params(colors="#444466", labelsize=5)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor("#222244")
        ax.grid(True, color="#222244", lw=0.4, alpha=0.6)
        ax.view_init(elev=28, azim=-65)
        ax.add_collection3d(Poly3DCollection([PALM], alpha=0.12,
                            facecolor="#5577aa", edgecolor="#7799cc", lw=0.7))

    arts_gt = _init_hand(ax_gt, _hand_fk(gt_seg[0]))
    arts_pr = _init_hand(ax_pr, _hand_fk(pred_seg[0]))

    fig.text(0.5, 0.96, f"MyoDigits -{run_name}",
             ha="center", color="white", fontsize=14, fontweight="bold")
    mae_txt = fig.text(0.5, 0.02, "", ha="center", color="#aaaacc", fontsize=10)

    fig.legend(handles=[
        Line2D([0],[0], color="#60a5fa", lw=3, label="Ground truth"),
        Line2D([0],[0], color="#fb923c", lw=3, label=f"Prediction"),
    ], loc="lower center", ncol=2, framealpha=0.2, facecolor="#222233",
       edgecolor="#555577", labelcolor="white", fontsize=10,
       bbox_to_anchor=(0.5, 0.06))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.12, wspace=0.04)

    def animate(i):
        _update_hand(arts_gt, _hand_fk(gt_seg[i]))
        _update_hand(arts_pr, _hand_fk(pred_seg[i]))
        mae_txt.set_text(
            f"Frame {i+1}/{n_frames}   |   "
            f"MAE this frame: {mae_per_frame[i]:.1f}°   |   "
            f"Test-set mean MAE: {mae:.3f}"
        )
        return []

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    writer = FFMpegWriter(fps=20, bitrate=2500,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    print("Rendering hand animation…")
    anim.save(str(out_path), writer=writer, dpi=130,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"Saved: {out_path}")


def _save_exercise_animation(segments, run_name, out_path, n_frames=200):
    """
    segments : list of (gt_seg, pred_seg, label) where each array is (>=n_frames, 20)
    Shows each segment sequentially with a label.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # trim / pick most-variable window within each segment
    clipped = []
    for gt, pred, label in segments:
        if len(gt) <= n_frames:
            clipped.append((gt, pred, label))
        else:
            step = 50
            scores = [gt[i:i+n_frames].var() for i in range(0, len(gt) - n_frames, step)]
            start  = int(np.argmax(scores)) * step
            clipped.append((gt[start:start+n_frames], pred[start:start+n_frames], label))

    total_frames = len(clipped) * n_frames

    BG = "#0f0f1a"; PANEL = "#151525"
    fig = plt.figure(figsize=(15, 7.5), facecolor=BG)
    ax_gt = fig.add_subplot(121, projection="3d", facecolor=PANEL)
    ax_pr = fig.add_subplot(122, projection="3d", facecolor=PANEL)

    for ax, lbl in [(ax_gt, "Ground Truth"), (ax_pr, "Prediction")]:
        ax.set_title(lbl, color="white", fontsize=13, pad=6, fontweight="bold")
        ax.set_xlim(-80, 80); ax.set_ylim(-20, 175); ax.set_zlim(-80, 60)
        ax.set_xlabel("X", color="#555577", fontsize=7, labelpad=2)
        ax.set_ylabel("Y", color="#555577", fontsize=7, labelpad=2)
        ax.set_zlabel("Z", color="#555577", fontsize=7, labelpad=2)
        ax.tick_params(colors="#444466", labelsize=5)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor("#222244")
        ax.grid(True, color="#222244", lw=0.4, alpha=0.6)
        ax.view_init(elev=28, azim=-65)
        ax.add_collection3d(Poly3DCollection([PALM], alpha=0.12,
                            facecolor="#5577aa", edgecolor="#7799cc", lw=0.7))

    gt0, pred0, _ = clipped[0]
    arts_gt = _init_hand(ax_gt, _hand_fk(gt0[0]))
    arts_pr = _init_hand(ax_pr, _hand_fk(pred0[0]))

    title_txt = fig.text(0.5, 0.96, f"MyoDigits - {run_name}",
                         ha="center", color="white", fontsize=14, fontweight="bold")
    seg_txt   = fig.text(0.5, 0.91, "", ha="center", color="#ccccee", fontsize=11)
    mae_txt   = fig.text(0.5, 0.02, "", ha="center", color="#aaaacc", fontsize=10)

    fig.legend(handles=[
        Line2D([0],[0], color="#60a5fa", lw=3, label="Ground truth"),
        Line2D([0],[0], color="#fb923c", lw=3, label="Prediction"),
    ], loc="lower center", ncol=2, framealpha=0.2, facecolor="#222233",
       edgecolor="#555577", labelcolor="white", fontsize=10,
       bbox_to_anchor=(0.5, 0.06))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12, wspace=0.04)

    def animate(i):
        seg_idx  = i // n_frames
        frame    = i %  n_frames
        gt_seg, pred_seg, label = clipped[seg_idx]
        mae_frame = float(np.abs(gt_seg[frame] - pred_seg[frame]).mean())

        _update_hand(arts_gt, _hand_fk(gt_seg[frame]))
        _update_hand(arts_pr, _hand_fk(pred_seg[frame]))
        seg_txt.set_text(label)
        mae_txt.set_text(
            f"Clip {seg_idx+1}/{len(clipped)}  |  "
            f"Frame {frame+1}/{n_frames}  |  "
            f"MAE this frame: {mae_frame:.1f}°"
        )
        return []

    anim = FuncAnimation(fig, animate, frames=total_frames, interval=50, blit=False)
    writer = FFMpegWriter(fps=20, bitrate=2500,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    print(f"Rendering {len(clipped)}-clip exercise animation ({total_frames} frames)…")
    anim.save(str(out_path), writer=writer, dpi=130,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── notebook ──────────────────────────────────────────────────────────────────
def _save_notebook(exp_dir, run_name, metrics, b64_training, b64_per_joint, anim_path):
    cells = []

    # header
    cells.append(new_markdown_cell(source=f"# {run_name} -Evaluation Results"))

    # metrics summary
    md_metrics = (
        f"## Metrics\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Test MAE | **{metrics['test_mae']:.4f}** |\n"
        f"| Test RMSE | {metrics['test_rmse']:.4f} |\n"
        f"| Test R² | {metrics['test_r2']:.4f} |\n"
        f"| Mean Pearson r | {metrics['mean_pearson_r']:.4f} |\n"
        f"| Test windows | {metrics['n_test_windows']:,} |\n"
    )
    cells.append(new_markdown_cell(source=md_metrics))

    # training curves
    cells.append(new_markdown_cell(source="## Training Curves"))
    cells.append(_image_output_cell(b64_training, "training curves"))

    # per-joint
    cells.append(new_markdown_cell(source="## Per-joint Metrics"))
    cells.append(_image_output_cell(b64_per_joint, "per-joint metrics"))

    # hand animation
    cells.append(new_markdown_cell(
        source=f"## 3-D Hand Animation\n\n"
               f"Saved to: `{anim_path.relative_to(exp_dir.parent)}`\n\n"
               f"Open with any video player, or run the cell below in a Jupyter environment."
    ))
    cells.append(new_code_cell(
        source=(
            f"from IPython.display import Video\n"
            f"Video('{anim_path}', embed=True, width=900)"
        )
    ))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb_path = exp_dir / "results.ipynb"
    nb_path.write_text(nbformat.writes(nb))
    print(f"Saved: {nb_path}")
