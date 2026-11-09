"""Lightweight CNN classifier for authentic vs AI-generated images.

Architecture (CPU-friendly, ~0.5 M parameters at 128x128 input):

  Conv(3->16, 3x3) -> ReLU -> MaxPool
  Conv(16->32, 3x3) -> ReLU -> MaxPool
  Conv(32->64, 3x3) -> ReLU -> MaxPool
  Global Average Pool
  Linear(64 -> 32) -> ReLU -> Dropout
  Linear(32 -> 1)   (logit for binary classification)

Total parameters are small enough to train on an i5 CPU in minutes on a few
thousand images, and inference is fast enough for interactive use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class LightweightCNN(nn.Module):
    def __init__(self, input_size: int = 128, num_classes: int = 1):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # After three 2x2 pools: 128 -> 16 spatial, 64 channels
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logit

    @property
    def feature_dim(self) -> int:
        return 64


def save_model(model: LightweightCNN, path: Path | str, metadata: Optional[dict] = None) -> None:
    """Save model weights plus optional metadata (input size, labels, etc.)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "input_size": model.input_size,
        "num_classes": model.num_classes,
    }
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_model(path: Path | str, device: Optional[torch.device] = None) -> LightweightCNN:
    """Load a saved classifier and return it in eval mode on the given device."""
    if device is None:
        device = torch.device("cpu")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    payload = torch.load(path, map_location=device)
    model = LightweightCNN(
        input_size=payload.get("input_size", 128),
        num_classes=payload.get("num_classes", 1),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model
