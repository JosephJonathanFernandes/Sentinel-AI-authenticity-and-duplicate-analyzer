"""Single-image and batch inference for the AI-detection classifier.

Results carry an explicit ``uncertain`` band: when model confidence is below
``uncertainty_threshold`` the label is reported as ``uncertain`` rather than
forcing a binary guess.  This keeps with the engineering rule that AI
detection must never claim 100% certainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from .classifier import LightweightCNN, load_model
from .preprocessing import get_eval_transform

LABEL_AUTHENTIC = "authentic"
LABEL_AI_GENERATED = "ai_generated"
LABEL_UNCERTAIN = "uncertain"


@dataclass
class PredictionResult:
    path: str
    label: str                     # "authentic" | "ai_generated" | "uncertain"
    confidence: float              # 0..1 probability of the chosen class
    ai_probability: float          # 0..1 probability the image is AI-generated
    uncertain: bool                # True if confidence < uncertainty_threshold
    model_version: Optional[str] = None
    error: Optional[str] = None


class AIClassifier:
    """Wrapper around the trained CNN for local inference."""

    def __init__(
        self,
        model_path: Path | str,
        image_size: int = 128,
        uncertainty_threshold: float = 0.60,
        device: Optional[torch.device] = None,
    ):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device or torch.device("cpu")
        self.model: Optional[LightweightCNN] = None
        self.transform = get_eval_transform(image_size)

    def load(self) -> None:
        """Load the model weights from disk."""
        self.model = load_model(self.model_path, device=self.device)

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load()

    def predict_path(self, path: Path | str) -> PredictionResult:
        """Classify a single image file by path."""
        self._ensure_loaded()
        path = Path(path)

        result = PredictionResult(
            path=str(path),
            label=LABEL_UNCERTAIN,
            confidence=0.0,
            ai_probability=0.5,
            uncertain=True,
            error=None,
        )

        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError, FileNotFoundError) as exc:
            result.error = f"Unable to load image: {exc}"
            return result

        try:
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit = self.model(tensor)
                prob = float(torch.sigmoid(logit).item())

            ai_probability = prob
            authentic_probability = 1.0 - prob

            if ai_probability >= authentic_probability:
                predicted_label = LABEL_AI_GENERATED
                confidence = ai_probability
            else:
                predicted_label = LABEL_AUTHENTIC
                confidence = authentic_probability

            result.ai_probability = ai_probability
            result.confidence = confidence
            result.uncertain = confidence < self.uncertainty_threshold
            result.label = (
                LABEL_UNCERTAIN if result.uncertain else predicted_label
            )
        except Exception as exc:  # pragma: no cover - defensive
            result.error = f"Inference failed: {exc}"

        return result

    def predict_batch(
        self,
        paths: list[Path | str],
        batch_size: int = 32,
    ) -> list[PredictionResult]:
        """Classify a list of images, streaming in bounded batches."""
        self._ensure_loaded()
        results: list[PredictionResult] = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            tensors = []
            valid_indices = []
            for j, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.transform(img))
                    valid_indices.append(j)
                except (UnidentifiedImageError, OSError, FileNotFoundError) as exc:
                    results.append(
                        PredictionResult(
                            path=str(p),
                            label=LABEL_UNCERTAIN,
                            confidence=0.0,
                            ai_probability=0.5,
                            uncertain=True,
                            error=f"Unable to load image: {exc}",
                        )
                    )

            if tensors:
                batch_tensor = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    logits = self.model(batch_tensor)
                    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                for local_idx, prob in zip(valid_indices, probs):
                    p = batch_paths[local_idx]
                    ai_probability = float(prob)
                    authentic_probability = 1.0 - ai_probability
                    if ai_probability >= authentic_probability:
                        predicted_label = LABEL_AI_GENERATED
                        confidence = ai_probability
                    else:
                        predicted_label = LABEL_AUTHENTIC
                        confidence = authentic_probability
                    uncertain = confidence < self.uncertainty_threshold
                    results.append(
                        PredictionResult(
                            path=str(p),
                            label=LABEL_UNCERTAIN if uncertain else predicted_label,
                            confidence=confidence,
                            ai_probability=ai_probability,
                            uncertain=uncertain,
                        )
                    )
        return results
