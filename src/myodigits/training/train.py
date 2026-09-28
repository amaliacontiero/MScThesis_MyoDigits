import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def train(model, loader_train, loader_val, config, device):

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epochs        = int(config["epochs"])
    lr            = float(config["lr"])
    patience      = int(config.get("patience", 10))
    weight_decay  = float(config.get("weight_decay", 1e-4))
    smooth_lambda = float(config.get("smooth_lambda", 0.0))
    warmup_epochs = int(config.get("warmup_epochs", 5))
    save_history  = bool(config.get("save_history", True))

    best_ckpt_path = checkpoint_dir / "best.pt"
    last_ckpt_path = checkpoint_dir / "last.pt"
    history_path   = checkpoint_dir / "history.json"

    opt_name  = config.get("optimizer", "adamw").lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()

    # Warmup (linear) + cosine decay scheduler, stepped every batch
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

    best_val_mae    = float("inf")
    best_epoch      = 0
    epochs_no_improve = 0
    history         = []
    global_step     = 0

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_items    = 0

        for Xb, yb in loader_train:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(Xb)

                if pred.dim() == 3:          # seq model: (B, T, 20)
                    pred_f = pred.reshape(-1, pred.shape[-1])
                    yb_f   = yb.reshape(-1, yb.shape[-1])
                else:
                    pred_f, yb_f = pred, yb

                loss = criterion(pred_f, yb_f)

                if smooth_lambda > 0.0 and pred.dim() == 3:
                    smooth = torch.abs(pred[:, 1:] - pred[:, :-1]).mean()
                    loss   = loss + smooth_lambda * smooth

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            n = pred_f.shape[0]
            train_loss_sum += loss.item() * n
            train_items    += n

        train_loss = train_loss_sum / max(train_items, 1)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_mae_sum = 0.0
        val_items   = 0

        with torch.no_grad():
            for Xb, yb in loader_val:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(Xb)

                if pred.dim() == 3:
                    pred = pred.reshape(-1, pred.shape[-1])
                    yb   = yb.reshape(-1, yb.shape[-1])

                val_mae_sum += (pred - yb).abs().mean(dim=1).sum().item()
                val_items   += pred.shape[0]

        val_mae    = val_mae_sum / max(val_items, 1)
        current_lr = scheduler.get_last_lr()[0]

        record = {
            "epoch":      epoch,
            "train_loss": float(train_loss),
            "val_mae":    float(val_mae),
            "lr":         float(current_lr),
        }
        history.append(record)

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae     = val_mae
            best_epoch       = epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mae":        best_val_mae,
                    "config":              dict(config),
                    "history":             history,
                },
                best_ckpt_path,
            )
            tag = " <--- BEST"
        else:
            epochs_no_improve += 1
            tag = f" (no improvement {epochs_no_improve}/{patience})"

        torch.save(
            {
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae":        best_val_mae,
                "config":              dict(config),
                "history":             history,
            },
            last_ckpt_path,
        )

        if save_history:
            history_path.write_text(json.dumps(history, indent=2))

        elapsed = time.time() - t_epoch
        print(
            f"Epoch {epoch:02d}/{epochs:02d} -- "
            f"train_loss={train_loss:.4f} -- "
            f"val_mae={val_mae:.4f} -- "
            f"lr={current_lr:.2e} -- "
            f"{elapsed:.1f}s{tag}"
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    print(f"\nBest val MAE: {best_val_mae:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")

    return best_val_mae, history
