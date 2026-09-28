"""
Episodic training for subject-conditional models (PoseNetSC).

Each batch carries (support, query_X, query_y) tuples from the same user.
The subject encoder produces z from support; the pose decoder predicts query
joint angles conditioned on z. End-to-end with backprop through both stages.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def train_sc(model, loader_train, loader_val, config, device):
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epochs        = int(config["epochs"])
    lr            = float(config["lr"])
    patience      = int(config.get("patience", 10))
    weight_decay  = float(config.get("weight_decay", 1e-4))
    smooth_lambda = float(config.get("smooth_lambda", 0.0))
    warmup_epochs = int(config.get("warmup_epochs", 5))

    best_ckpt_path = checkpoint_dir / "best.pt"
    last_ckpt_path = checkpoint_dir / "last.pt"
    history_path   = checkpoint_dir / "history.json"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()

    total_steps  = epochs * len(loader_train)
    warmup_steps = warmup_epochs * len(loader_train)

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_mae      = float("inf")
    best_epoch        = 0
    epochs_no_improve = 0
    history           = []

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss_sum, train_items = 0.0, 0

        for Xs, Xq, yq in loader_train:
            Xs = Xs.to(device, non_blocking=True)
            Xq = Xq.to(device, non_blocking=True)
            yq = yq.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(Xq, support=Xs)            # (B, T, 20)
                pred_f = pred.reshape(-1, pred.shape[-1])
                yq_f   = yq.reshape(-1, yq.shape[-1])
                loss   = criterion(pred_f, yq_f)
                if smooth_lambda > 0.0:
                    smooth = torch.abs(pred[:, 1:] - pred[:, :-1]).mean()
                    loss   = loss + smooth_lambda * smooth

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            n = pred_f.shape[0]
            train_loss_sum += loss.item() * n
            train_items    += n

        train_loss = train_loss_sum / max(train_items, 1)

        # ── validate (still episodic; support resampled per item) ──────────
        model.eval()
        val_mae_sum, val_items = 0.0, 0
        with torch.no_grad():
            for Xs, Xq, yq in loader_val:
                Xs = Xs.to(device, non_blocking=True)
                Xq = Xq.to(device, non_blocking=True)
                yq = yq.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(Xq, support=Xs)

                pred_f = pred.reshape(-1, pred.shape[-1])
                yq_f   = yq.reshape(-1, yq.shape[-1])
                val_mae_sum += (pred_f - yq_f).abs().mean(dim=1).sum().item()
                val_items   += pred_f.shape[0]

        val_mae    = val_mae_sum / max(val_items, 1)
        current_lr = scheduler.get_last_lr()[0]

        record = {"epoch": epoch, "train_loss": float(train_loss),
                  "val_mae": float(val_mae), "lr": float(current_lr)}
        history.append(record)

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae      = val_mae
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae":         best_val_mae,
                "config":               dict(config),
                "history":              history,
            }, best_ckpt_path)
            tag = " <--- BEST"
        else:
            epochs_no_improve += 1
            tag = f" (no improvement {epochs_no_improve}/{patience})"

        torch.save({
            "epoch": epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mae":         best_val_mae,
            "config":               dict(config),
            "history":              history,
        }, last_ckpt_path)
        history_path.write_text(json.dumps(history, indent=2))

        elapsed = time.time() - t_epoch
        print(f"Epoch {epoch:02d}/{epochs:02d} -- train_loss={train_loss:.4f} -- "
              f"val_mae={val_mae:.4f} -- lr={current_lr:.2e} -- {elapsed:.1f}s{tag}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    print(f"\nBest val MAE: {best_val_mae:.4f} at epoch {best_epoch}")
    return best_val_mae, history
