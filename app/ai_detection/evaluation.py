"""Evaluation metrics for the AI-detection classifier.

Computes accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and
false-positive / false-negative rates.  All metrics are computed from model
probabilities (not hard labels) where possible so thresholds can be varied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .classifier import LightweightCNN, load_model
from .preprocessing import get_eval_transform
from .dataset import LABEL_AUTHENTIC, LABEL_SYNTHETIC


@dataclass
class EvaluationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]]
    false_positive_rate: float
    false_negative_rate: float
    n_samples: int
    n_authentic: int
    n_ai_generated: int
    threshold: float

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "n_samples": self.n_samples,
            "n_authentic": self.n_authentic,
            "n_ai_generated": self.n_ai_generated,
            "threshold": self.threshold,
        }


def evaluate_from_loader(
    model: LightweightCNN,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> EvaluationReport:
    """Evaluate a trained model over a PyTorch DataLoader of (img, label)."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for imgs, targets, _paths, _corrupt in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).squeeze(1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(targets.cpu().numpy().astype(int).tolist())

    return evaluate_from_arrays(np.array(all_probs), np.array(all_labels), threshold=threshold)


def evaluate_from_arrays(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationReport:
    """Compute metrics from probability / label arrays."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    n_samples = int(len(labels))
    n_authentic = int((labels == LABEL_AUTHENTIC).sum())
    n_ai_generated = int((labels == LABEL_SYNTHETIC).sum())

    preds = (probs >= threshold).astype(int)

    accuracy = float(accuracy_score(labels, preds))
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))

    cm = confusion_matrix(labels, preds, labels=[LABEL_AUTHENTIC, LABEL_SYNTHETIC])
    # cm = [[TN, FP], [FN, TP]] where class 0 = authentic, class 1 = ai_generated
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    fpr = fp / max(fp + tn, 1)  # authentic flagged as AI
    fnr = fn / max(fn + tp, 1)  # AI flagged as authentic

    try:
        roc_auc_val = float(roc_auc_score(labels, probs))
        roc_auc = roc_auc_val if not np.isnan(roc_auc_val) else None
    except ValueError:
        roc_auc = None  # only one class present

    return EvaluationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        false_positive_rate=float(fpr),
        false_negative_rate=float(fnr),
        n_samples=n_samples,
        n_authentic=n_authentic,
        n_ai_generated=n_ai_generated,
        threshold=threshold,
    )


def evaluate_model_on_path(
    model: LightweightCNN,
    image_paths: list,
    labels: list,
    device: torch.device,
    image_size: int = 128,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> EvaluationReport:
    """Convenience: evaluate over a list of (path, label) pairs."""
    from .preprocessing import ImagePathDataset
    from torch.utils.data import DataLoader

    ds = ImagePathDataset(
        image_paths, labels, transform=get_eval_transform(image_size)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return evaluate_from_loader(model, loader, device, threshold=threshold)
