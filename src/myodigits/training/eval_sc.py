"""
Evaluation wrapper for subject-conditional models.

Calibrates the model from K unlabeled windows of the test subject, caches the
embedding inside the model, then delegates to the standard `run_eval` (which
calls model(Xb) and benefits from the cached z via PoseNetSC's forward
fallback path).

Also records the calibration size in metrics for the per-K sweep experiments.
"""

from pathlib import Path

import torch

from myodigits.training.eval import run_eval


def run_eval_sc(model, loader_test, calib_windows, history, config, device,
                exp_dir, run_name, jnt_denorm=None):
    """
    Parameters
    ----------
    calib_windows : tensor (K, C, 400) or (1, K, C, 400) — unlabeled windows
                    from the test subject. Used once for calibration.
    """
    # Load best checkpoint *before* calibration so the encoder used for z is
    # the best one. run_eval will reload the same checkpoint internally; the
    # _z_cache buffer is non-persistent so it is preserved through that load.
    ckpt_dir  = Path(config["checkpoint_dir"])
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[SC] Loaded best ckpt (epoch {ckpt['epoch']}, "
              f"val_mae {ckpt['best_val_mae']:.4f})")

    model.eval()
    cw = calib_windows.to(device)
    if cw.dim() == 3:
        cw = cw.unsqueeze(0)
    model.calibrate(cw)
    K = cw.shape[1]
    print(f"[SC] Calibrated subject embedding from {K} unlabeled windows "
          f"(~{K * 50 / 1000:.1f} s @ 50 ms hop)")

    metrics = run_eval(model, loader_test, history, config, device,
                       exp_dir, run_name, jnt_denorm=jnt_denorm)
    metrics["n_calibration_windows"] = K
    return metrics
