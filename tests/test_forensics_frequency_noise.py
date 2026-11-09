"""Tests for frequency (FFT/DCT) and noise forensic analysis.

Verifies features are computed without crashing and that corrupted/missing
files degrade gracefully.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from sentinel.app.forensics.frequency_analysis import analyze_frequency, FrequencyFeatures
from sentinel.app.forensics.noise_analysis import analyze_noise, NoiseFeatures


def _save_noisy_image(path: Path, noise_level: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:] = 128
    rng = np.random.default_rng(7)
    noise = (rng.normal(0, noise_level * 255, arr.shape)).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")


def test_frequency_features_computed(tmp_path):
    p = tmp_path / "freq.png"
    _save_noisy_image(p)

    feats = analyze_frequency(p, fft_size=64)

    assert feats.error is None
    assert feats.fft_energy_ratio_low is not None
    assert feats.fft_energy_ratio_mid is not None
    assert feats.fft_energy_ratio_high is not None
    # Ratios should sum to approximately 1.
    total = (
        feats.fft_energy_ratio_low
        + feats.fft_energy_ratio_mid
        + feats.fft_energy_ratio_high
    )
    assert abs(total - 1.0) < 0.02, f"FFT band ratios sum to {total}"
    assert 0.0 <= feats.fft_spectral_flatness <= 1.0
    assert feats.dct_energy_ratio_low is not None
    assert feats.compression_indicator in (None, "none_or_mild", "moderate_blocking_artifacts", "strong_blocking_artifacts")


def test_frequency_nonexistent_file(tmp_path):
    p = tmp_path / "missing.png"
    feats = analyze_frequency(p, fft_size=64)
    assert feats.error is not None


def test_frequency_corrupt_file(tmp_path):
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"garbage")
    feats = analyze_frequency(p, fft_size=64)
    assert feats.error is not None


def test_noise_features_computed(tmp_path):
    p = tmp_path / "noise.png"
    _save_noisy_image(p, noise_level=0.15)

    nf = analyze_noise(p, sigma=1.0)

    assert nf.error is None
    assert nf.noise_residual_std is not None
    assert nf.noise_residual_std >= 0.0
    assert nf.noise_local_variance_mean is not None
    assert nf.noise_local_variance_mean >= 0.0
    assert 0.0 <= nf.noise_consistency_score <= 1.0


def test_noise_over_smooth_image_low_std(tmp_path):
    """A heavily blurred image should produce a very low residual std."""
    p = tmp_path / "smooth.png"
    Image.new("RGB", (128, 128), (128, 128, 128)).save(p, format="PNG")

    nf = analyze_noise(p, sigma=1.0)
    assert nf.error is None
    # Uniform image -> after median filtering, residual ~ 0.
    assert nf.noise_residual_std is not None
    assert nf.noise_residual_std < 0.01


def test_noise_nonexistent_file(tmp_path):
    p = tmp_path / "missing.png"
    nf = analyze_noise(p)
    assert nf.error is not None


def test_frequency_deterministic(tmp_path):
    p = tmp_path / "det.png"
    _save_noisy_image(p)
    a = analyze_frequency(p, fft_size=64)
    b = analyze_frequency(p, fft_size=64)
    assert a.fft_energy_ratio_low == b.fft_energy_ratio_low
    assert a.fft_energy_ratio_high == b.fft_energy_ratio_high
