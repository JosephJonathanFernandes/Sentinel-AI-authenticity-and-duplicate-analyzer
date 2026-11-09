"""Frequency-domain forensic features: FFT and DCT analysis.

These features look for anomalies in the frequency spectrum that can be
characteristic of AI-generated images (e.g. unusual high-frequency energy,
regular grid patterns from upsampling, or missing camera-sensor noise
spectra).  All computations are CPU-based and memory-bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import fft as sp_fft

from ..core.image_loader import ImageLoadError, load_image, close_image


@dataclass
class FrequencyFeatures:
    """Results of frequency-domain analysis."""

    path: str
    fft_energy_ratio_low: Optional[float] = None
    fft_energy_ratio_mid: Optional[float] = None
    fft_energy_ratio_high: Optional[float] = None
    fft_spectral_flatness: Optional[float] = None
    dct_energy_ratio_low: Optional[float] = None
    dct_energy_ratio_mid: Optional[float] = None
    dct_energy_ratio_high: Optional[float] = None
    compression_indicator: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "fft_energy_ratio_low": self.fft_energy_ratio_low,
            "fft_energy_ratio_mid": self.fft_energy_ratio_mid,
            "fft_energy_ratio_high": self.fft_energy_ratio_high,
            "fft_spectral_flatness": self.fft_spectral_flatness,
            "dct_energy_ratio_low": self.dct_energy_ratio_low,
            "dct_energy_ratio_mid": self.dct_energy_ratio_mid,
            "dct_energy_ratio_high": self.dct_energy_ratio_high,
            "compression_indicator": self.compression_indicator,
            "error": self.error,
        }


def _image_to_gray_float(path: Path) -> Optional[np.ndarray]:
    """Load an image and return a float32 grayscale array (0..1)."""
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


def _radial_bands(shape: tuple[int, int], n_bands: int = 3) -> np.ndarray:
    """Build a radial band mask for an FFT magnitude spectrum.

    Returns an integer array of the same shape where each pixel is labelled
    0 (low), 1 (mid), or 2 (high) frequency based on distance from centre.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = np.sqrt(cy ** 2 + cx ** 2)
    if max_radius == 0:
        return np.zeros(shape, dtype=int)
    normalised = radius / max_radius
    bands = np.floor(normalised * n_bands).astype(int)
    bands = np.clip(bands, 0, n_bands - 1)
    return bands


def _spectral_flatness(magnitude: np.ndarray) -> float:
    """Spectral flatness = geometric mean / arithmetic mean of magnitude.

    Values near 1 indicate noise-like (flat) spectrum; values near 0 indicate
    tonal/peaky spectrum.  Camera images often show some flatness from sensor
    noise; AI images can be either flatter or peakier depending on generator.
    """
    mag = magnitude + 1e-10  # avoid log(0)
    geom_mean = np.exp(np.mean(np.log(mag)))
    arith_mean = np.mean(mag)
    if arith_mean == 0:
        return 0.0
    return float(geom_mean / arith_mean)


def analyze_frequency(path: Path | str, fft_size: int = 256) -> FrequencyFeatures:
    """Compute FFT and DCT forensic features for an image.

    Parameters
    ----------
    path : str | Path
        Image file path.
    fft_size : int
        The image is resized to (fft_size, fft_size) before analysis to bound
        computation and make results comparable across images.
    """
    result = FrequencyFeatures(path=str(path))
    arr = _image_to_gray_float(Path(path))
    if arr is None:
        result.error = "Unable to load image for frequency analysis"
        return result

    try:
        # Resize to a fixed size for consistent, bounded computation.
        from PIL import Image as PILImage

        img = PILImage.fromarray((arr * 255).astype(np.uint8)).resize(
            (fft_size, fft_size), PILImage.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # ---- FFT analysis ----
        f_transform = sp_fft.fft2(arr)
        f_shift = sp_fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        total_energy = float(np.sum(magnitude))
        bands = _radial_bands(arr.shape, n_bands=3)

        if total_energy > 0:
            for band_idx in range(3):
                band_energy = float(np.sum(magnitude[bands == band_idx]))
                ratio = band_energy / total_energy
                if band_idx == 0:
                    result.fft_energy_ratio_low = ratio
                elif band_idx == 1:
                    result.fft_energy_ratio_mid = ratio
                else:
                    result.fft_energy_ratio_high = ratio

        result.fft_spectral_flatness = _spectral_flatness(magnitude)

        # ---- DCT analysis ----
        dct = sp_fft.dctn(arr, type=2, norm="ortho")
        dct_magnitude = np.abs(dct)
        dct_total = float(np.sum(dct_magnitude))
        if dct_total > 0:
            for band_idx in range(3):
                band_energy = float(np.sum(dct_magnitude[bands == band_idx]))
                ratio = band_energy / dct_total
                if band_idx == 0:
                    result.dct_energy_ratio_low = ratio
                elif band_idx == 1:
                    result.dct_energy_ratio_mid = ratio
                else:
                    result.dct_energy_ratio_high = ratio

        # ---- Compression indicator ----
        result.compression_indicator = _estimate_compression(arr, magnitude)

    except Exception as exc:  # pragma: no cover - defensive
        result.error = f"Frequency analysis failed: {exc}"

    return result


def _estimate_compression(
    arr: np.ndarray, magnitude: np.ndarray
) -> Optional[str]:
    """Heuristic JPEG/blocking-artifact indicator from the FFT spectrum.

    JPEG compression introduces periodic peaks at DCT block frequencies (8x8
    blocks).  We look for energy concentration at those radial frequencies.
    Returns a qualitative label or None.
    """
    try:
        h, w = arr.shape
        cy, cx = h // 2, w // 2
        # 8x8 block frequency corresponds to radius ~ (h/8) cycles in each axis.
        block_radius = max(h, w) / 8.0
        yy, xx = np.mgrid[0:h, 0:w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = (radius > block_radius * 0.7) & (radius < block_radius * 1.3)
        if not np.any(mask):
            return None
        block_energy = float(np.sum(magnitude[mask]))
        total = float(np.sum(magnitude))
        if total == 0:
            return None
        ratio = block_energy / total
        if ratio > 0.05:
            return "strong_blocking_artifacts"
        if ratio > 0.02:
            return "moderate_blocking_artifacts"
        return "none_or_mild"
    except Exception:
        return None
