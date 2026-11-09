"""Forensic engine: orchestrates metadata + frequency + noise analysis.

Produces a combined ForensicReport for one image.  All signals are treated as
*supporting* forensic evidence — none is definitive on its own, and missing
or unusual metadata does NOT prove an image is AI-generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .metadata import ImageMetadata, extract_metadata
from .frequency_analysis import FrequencyFeatures, analyze_frequency
from .noise_analysis import NoiseFeatures, analyze_noise
from ..config import settings


@dataclass
class ForensicReport:
    """Combined forensic report for a single image."""

    path: str
    file_name: str
    directory: str
    extension: str
    size_bytes: int

    metadata: Optional[ImageMetadata] = None
    frequency: Optional[FrequencyFeatures] = None
    noise: Optional[NoiseFeatures] = None

    # Composite signals (0..1 where meaningful)
    metadata_suspicious: bool = False
    frequency_anomaly_score: float = 0.0   # higher = more frequency anomaly
    noise_anomaly_score: float = 0.0       # higher = more noise anomaly

    # Overall supporting-evidence score (0..1, higher = more forensic anomalies)
    composite_score: float = 0.0

    # Errors from individual stages (non-fatal)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "file_name": self.file_name,
            "directory": self.directory,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata.__dict__ if self.metadata else None,
            "frequency": self.frequency.as_dict() if self.frequency else None,
            "noise": self.noise.as_dict() if self.noise else None,
            "metadata_suspicious": self.metadata_suspicious,
            "frequency_anomaly_score": self.frequency_anomaly_score,
            "noise_anomaly_score": self.noise_anomaly_score,
            "composite_score": self.composite_score,
            "errors": self.errors,
        }


class ForensicEngine:
    """Runs the full forensic pipeline on one image file."""

    def __init__(self, fft_size: Optional[int] = None, noise_sigma: Optional[float] = None):
        self.fft_size = fft_size or settings.fft_size
        self.noise_sigma = noise_sigma or settings.noise_sigma

    def analyze(self, path: Path | str) -> ForensicReport:
        """Run metadata, frequency, and noise analysis on `path`."""
        path = Path(path)
        report = ForensicReport(
            path=str(path),
            file_name=path.name,
            directory=str(path.parent),
            extension=path.suffix.lower(),
            size_bytes=path.stat().st_size if path.exists() else 0,
        )

        # ---- Metadata ----
        try:
            report.metadata = extract_metadata(path)
            if report.metadata.error:
                report.errors.append(f"metadata: {report.metadata.error}")
            report.metadata_suspicious = bool(report.metadata.suspicious)
        except Exception as exc:  # pragma: no cover - defensive
            report.errors.append(f"metadata: {exc}")

        # ---- Frequency ----
        try:
            report.frequency = analyze_frequency(path, fft_size=self.fft_size)
            if report.frequency.error:
                report.errors.append(f"frequency: {report.frequency.error}")
            report.frequency_anomaly_score = self._score_frequency(report.frequency)
        except Exception as exc:  # pragma: no cover - defensive
            report.errors.append(f"frequency: {exc}")

        # ---- Noise ----
        try:
            report.noise = analyze_noise(path, sigma=self.noise_sigma)
            if report.noise.error:
                report.errors.append(f"noise: {report.noise.error}")
            report.noise_anomaly_score = self._score_noise(report.noise)
        except Exception as exc:  # pragma: no cover - defensive
            report.errors.append(f"Noise: {exc}")

        # ---- Composite ----
        report.composite_score = self._composite(report)

        return report

    @staticmethod
    def _score_frequency(freq: Optional[FrequencyFeatures]) -> float:
        """Convert frequency features into a 0..1 anomaly score.

        Heuristics (supporting evidence only):
        * Very low high-frequency energy ratio can indicate over-smoothing.
        * Very low spectral flatness can indicate strong tonal peaks (artifacts).
        * Strong blocking artifacts can indicate heavy recompression.
        """
        if not freq:
            return 0.0
        score = 0.0
        if freq.fft_energy_ratio_high is not None and freq.fft_energy_ratio_high < 0.05:
            score += 0.3
        if freq.fft_spectral_flatness is not None and freq.fft_spectral_flatness < 0.1:
            score += 0.3
        if freq.compression_indicator == "strong_blocking_artifacts":
            score += 0.2
        elif freq.compression_indicator == "moderate_blocking_artifacts":
            score += 0.1
        return float(min(1.0, score))

    @staticmethod
    def _score_noise(noise: Optional[NoiseFeatures]) -> float:
        """Convert noise features into a 0..1 anomaly score.

        Heuristics (supporting evidence only):
        * Very low residual std -> over-smooth image.
        * Low noise consistency score -> spatially inconsistent noise.
        * High kurtosis in residual -> sparse outliers (possible artefacts).
        """
        if not noise:
            return 0.0
        score = 0.0
        if noise.noise_residual_std is not None and noise.noise_residual_std < 0.005:
            score += 0.35
        if noise.noise_consistency_score is not None and noise.noise_consistency_score < 0.4:
            score += 0.35
        if noise.noise_residual_kurtosis is not None and noise.noise_residual_kurtosis > 10:
            score += 0.3
        return float(min(1.0, score))

    def _composite(self, report: ForensicReport) -> float:
        """Weighted composite forensic anomaly score (0..1)."""
        components = [
            (report.metadata_suspicious, 0.2),
            (report.frequency_anomaly_score, 0.4),
            (report.noise_anomaly_score, 0.4),
        ]
        total = sum(val * w for val, w in components)
        return float(min(1.0, total))
