"""Grad-CAM explainability for the AI-detection CNN.

Produces class-activation heatmaps showing which image regions influenced
the model's prediction.  Heatmaps are computed from real gradients flowing
through the trained network — they are never fabricated.

Supports the engineering rule: AI detection explanations must be grounded
in actual model gradients, not placeholder visuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from .classifier import LightweightCNN, load_model
from .preprocessing import get_eval_transform


@dataclass
class GradCAMResult:
    """Grad-CAM output for a single image."""

    path: str
    heatmap: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    ai_probability: float = 0.5
    label: str = "uncertain"
    confidence: float = 0.0
    overlay: Optional[np.ndarray] = None  # (H, W, 3) uint8 BGR heatmap-overlaid image
    error: Optional[str] = None


class GradCAM:
    """Grad-CAM generator for the lightweight CNN.

    Hooks into the last convolutional layer of ``features`` to capture
    activation maps and gradients, then produces a class-activation map.
    """

    def __init__(
        self,
        model: LightweightCNN,
        target_layer: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        # Last conv layer inside the features Sequential block.
        # features = [Conv(0), BN(1), ReLU(2), Pool(3),
        #             Conv(4), BN(5), ReLU(6), Pool(7),
        #             Conv(8), BN(9), ReLU(10), Pool(11)]
        self.target_layer_name = target_layer or "features.8"
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        target = dict(self.model.named_modules()).get(self.target_layer_name)
        if target is None:
            raise ValueError(f"Target layer '{self.target_layer_name}' not found in model.")

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_full_backward_hook(backward_hook))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    @staticmethod
    def _compute_cam(
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> np.ndarray:
        """Compute the class-activation map from activations and gradients.

        Parameters
        ----------
        activations : (1, C, H, W)
        gradients   : (1, C, H, W)

        Returns
        -------
        np.ndarray : (H, W) float32 in [0, 1]
        """
        # Global average pool gradients over spatial dims -> (1, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activations -> (1, 1, H, W)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU: keep only positive influence
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.size > 0 and cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """Run Grad-CAM on a preprocessed image tensor.

        Parameters
        ----------
        image_tensor : (1, 3, H, W) already normalized, requires_grad=True
        target_class : 0 (authentic) or 1 (ai_generated).
            If None, uses the model's own prediction.

        Returns
        -------
        heatmap : (H, W) float32 in [0, 1]
        probability : float (sigmoid of logit)
        """
        self._activations = None
        self._gradients = None

        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(True)

        # Forward pass
        logit = self.model(image_tensor)  # (1, 1)
        probability = float(torch.sigmoid(logit).item())

        # Determine target class
        if target_class is None:
            target_class = 1 if probability >= 0.5 else 0

        # Backward pass on the selected class score.
        # Single logit output: score = logit for AI class, -logit for authentic.
        score = logit if target_class == 1 else -logit
        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        heatmap = self._compute_cam(self._activations, self._gradients)
        return heatmap, probability

    def close(self) -> None:
        self._remove_hooks()


def preprocess_for_gradcam(
    image: PILImage.Image,
    image_size: int = 128,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Preprocess a PIL image for Grad-CAM (no random augmentation)."""
    transform = get_eval_transform(image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    return tensor.to(device or torch.device("cpu"))


def overlay_heatmap(
    original: PILImage.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a heatmap on the original image using a jet colormap.

    Returns a uint8 (H, W, 3) BGR image (OpenCV format).
    """
    # Resize heatmap to original image size
    h, w = original.size[1], original.size[0]  # PIL: (w, h) -> cv2: (h, w)
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to uint8 jet colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Original image as BGR uint8
    original_arr = np.array(original.convert("RGB"))
    original_bgr = cv2.cvtColor(original_arr, cv2.COLOR_RGB2BGR)

    # Blend
    overlay = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay


class GradCAMExplainer:
    """High-level wrapper: load model, explain an image file, return results."""

    def __init__(
        self,
        model_path: Path | str,
        image_size: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.device = device or torch.device("cpu")
        self.model: Optional[LightweightCNN] = None
        self.gradcam: Optional[GradCAM] = None

    def load(self) -> None:
        self.model = load_model(self.model_path, device=self.device)
        self.gradcam = GradCAM(self.model, device=self.device)

    def explain_path(
        self,
        path: Path | str,
        alpha: float = 0.5,
    ) -> GradCAMResult:
        """Run Grad-CAM on an image file and return heatmap + overlay."""
        path = Path(path)
        result = GradCAMResult(
            path=str(path),
            heatmap=np.zeros((1, 1), dtype=np.float32),
            ai_probability=0.5,
            label="uncertain",
            confidence=0.0,
        )

        try:
            img = PILImage.open(path).convert("RGB")
        except Exception as exc:
            result.error = f"Unable to load image: {exc}"
            return result

        try:
            if self.model is None:
                self.load()
            if self.gradcam is None:
                raise RuntimeError("Grad-CAM not initialised")

            tensor = preprocess_for_gradcam(img, self.image_size, self.device)
            heatmap, probability = self.gradcam.explain(tensor)

            # Label / confidence
            authentic_prob = 1.0 - probability
            if probability >= authentic_prob:
                predicted_label = "ai_generated"
                confidence = probability
            else:
                predicted_label = "authentic"
                confidence = authentic_prob

            # Overlay heatmap on original image
            overlay = overlay_heatmap(img, heatmap, alpha=alpha)

            result.heatmap = heatmap
            result.ai_probability = probability
            result.label = predicted_label
            result.confidence = float(confidence)
            result.overlay = overlay
        except Exception as exc:  # pragma: no cover - defensive
            result.error = f"Grad-CAM failed: {exc}"

        return result

    def close(self) -> None:
        if self.gradcam is not None:
            self.gradcam.close()
