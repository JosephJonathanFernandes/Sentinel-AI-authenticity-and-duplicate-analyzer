"""Training loop for the lightweight AI-detection CNN.

Trains on CPU with early stopping based on validation loss.  Saves the best
checkpoint to disk.  Returns a small history dict so callers can report
learning curves if desired.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .classifier import LightweightCNN, save_model
from .preprocessing import create_data_loaders
from .dataset import DatasetSplit


def _binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        correct = (preds == targets.long()).sum().item()
        return correct / max(targets.numel(), 1)


def train_model(
    split: DatasetSplit,
    model_path: Path | str,
    *,
    image_size: int = 128,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-3,
    patience: int = 3,
    device: Optional[torch.device] = None,
    log_every: int = 1,
) -> dict:
    """Train the classifier and return a history dict.

    Parameters
    ----------
    split : DatasetSplit
        Train / val / test image lists.
    model_path : str | Path
        Where the best checkpoint is saved.
    image_size : int
        Input resolution (must match inference).
    batch_size : int
        Samples per batch.
    epochs : int
        Maximum training epochs.
    lr : float
        Adam learning rate.
    patience : int
        Early-stopping patience (epochs without val-loss improvement).
    device : torch.device | None
        Defaults to CPU.

    Returns
    -------
    dict
        Keys: ``best_val_loss``, ``epochs_run``, ``history`` (list of per-epoch
        dicts with train/val loss and accuracy).
    """
    if device is None:
        device = torch.device("cpu")

    loaders = create_data_loaders(split, image_size=image_size, batch_size=batch_size)
    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    if train_loader is None:
        raise ValueError("DatasetSplit must contain training images.")

    model = LightweightCNN(input_size=image_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = math.inf
    best_state: Optional[dict] = None
    patience_counter = 0
    history = []

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ---- training ----
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        for imgs, targets, _paths, _corrupt in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += _binary_accuracy(logits.detach(), targets.squeeze(1))
            train_batches += 1

        train_loss /= max(train_batches, 1)
        train_acc /= max(train_batches, 1)

        # ---- validation ----
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        model.eval()
        with torch.no_grad():
            if val_loader is not None:
                for imgs, targets, _paths, _corrupt in val_loader:
                    imgs = imgs.to(device)
                    targets = targets.to(device).unsqueeze(1).float()
                    logits = model(imgs)
                    loss = criterion(logits, targets)
                    val_loss += loss.item()
                    val_acc += _binary_accuracy(logits, targets.squeeze(1))
                    val_batches += 1
                val_loss /= max(val_batches, 1)
                val_acc /= max(val_batches, 1)
            else:
                val_loss = train_loss
                val_acc = train_acc

        epoch_time = time.time() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_seconds": epoch_time,
            }
        )

        if log_every and epoch % log_every == 0:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
                f"{epoch_time:.1f}s"
            )

        # ---- early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience}).")
                break

    # Restore best weights before saving.
    if best_state is not None:
        model.load_state_dict(best_state)

    save_model(model, model_path, metadata={"epochs_run": epoch, "best_val_loss": best_val_loss})

    return {
        "best_val_loss": best_val_loss,
        "epochs_run": epoch,
        "history": history,
    }
