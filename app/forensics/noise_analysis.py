"""Noise-residual forensic analysis.

Camera sensors introduce a characteristic noise pattern (shot noise + read
noise + fixed-pattern noise) that is spatially consistent across an image.
AI-generated images often show:

* Over-smooth or artificially uniform noise
* Noise that is inconsistent across local regions
* Noise patterns that match the generator's upsampling artefacts

We compute the noise residual (original minus a denoised version) and
summarise its statistics plus local consistency.  This is supporting
evidence only — not a definitive AI detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter, uniform_filter

from ..core.image_loader import ImageLoadError, load_image, close_image


@dataclass
class NoiseFeatures:
    """Noise-residual forensic features."""

    path: str
    noise_residual_std: Optional[float] = None
    noise_residual_mean: Optional[float] = None
    noise_residual_skew: Optional[float] = None
    noise_residual_kurtosis: Optional[float] = None
    noise_local_variance_mean: Optional[float] = None
    noise_local_variance_std: Optional[float] = None
    noise_consistency_score: Optional[float] = None  # 0..1, higher = more consistent
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "noise_residual_std": self.noise_residual_std,
            "noise_residual_mean": self.noise_residual_mean,
            "noise_residual_skew": self.noise_residual_skew,
            "noise_residual_kurtosis": self.noise_residual_kurtosis,
            "noise_local_variance_mean": self.noise_local_variance_mean,
            "noise_local_variance_std": self.noise_local_variance_std,
            "noise_consistency_score": self.noise_consistency_score,
            "error": self.error,
        }


def _image_to_gray_float(path: Path) -> Optional[np.ndarray]:
    try:
        loaded = load_image(path)
    except ImageLoadError:
        return None
    try:
        img = loaded.image.convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr
    except Exception:
        return None
    finally:
        close_image(loaded.image)


def _skew(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def analyze_noise(path: Path | str, sigma: float = 1.0) -> NoiseFeatures:
    """Compute noise-residual features for an image.

    Parameters
    ----------
    path : str | Path
        Image file path.
    sigma : float
        Denoising strength passed to the median filter kernel size
        (kernel = max(3, int(2*sigma)+1)).  Larger = stronger smoothing.
    """
    result = NoiseFeatures(path=str(path))
    arr = _image_to_gray_float(Path(path))
    if arr is None:
        result.error = "Unable to load image for noise analysis"
        return result

    try:
        # Denoise with a median filter (robust to edges, preserves structure).
        kernel = max(3, int(2 * sigma) + 1)
        if kernel % 2 == 0:
            kernel += 1
        denoised = median_filter(arr, size=kernel)

        # Noise residual = original - denoised.
        residual = arr - denoised

        # Global residual statistics.
        result.noise_residual_std = float(residual.std())
        result.noise_residual_mean = float(residual.mean())
        result.noise_residual_skew = _skew(residual)
        result.noise_residual_kurtosis = _kurtosis(residual)

        # Local variance of the residual (noise consistency map).
        local_mean = uniform_filter(residual, size=16)
        local_sq_mean = uniform_filter(residual * residual, size=16)
        local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
        result.noise_local_variance_mean = float(local_var.mean())
        result.noise_local_variance_std = float(local_var.std())

        # Consistency score: coefficient of variation of local variance.
        # Lower local-variance std relative to mean => more consistent noise.
        mean_var = result.noise_local_variance_mean or 0.0
        std_var = result.noise_local_variance_std or 0.0
        if mean_var > 0:
            cv = std_var / mean_var
            # Map CV to a 0..1 consistency score (inverted).
            result.noise_consistency_score = float(max(0.0, min(1.0, 1.0 - cv)))
        else:
            result.noise_consistency_score = 0.0

    except Exception as exc:  # pragma: no cover - defensive
        result.error = f"Noise analysis failed: {exc}"

    return result
