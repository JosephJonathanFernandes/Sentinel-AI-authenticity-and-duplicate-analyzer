"""Tests for the forensic engine orchestrator.

Verifies combined report generation, composite scoring, and error handling.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import numpy as np

from sentinel.app.forensics.forensic_engine import ForensicEngine, ForensicReport
from sentinel.app.forensics.metadata import ImageMetadata


def _save_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:] = 120
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 12, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")


def test_forensic_engine_full_run(tmp_path):
    p = tmp_path / "image.png"
    _save_image(p)

    engine = ForensicEngine(fft_size=64)
    report = engine.analyze(p)

    assert isinstance(report, ForensicReport)
    assert report.path == str(p)
    assert report.file_name == "image.png"
    assert report.extension == ".png"
    assert report.metadata is not None
    assert report.frequency is not None
    assert report.noise is not None
    # Composite score must be 0..1
    assert 0.0 <= report.composite_score <= 1.0
    assert 0.0 <= report.frequency_anomaly_score <= 1.0
    assert 0.0 <= report.noise_anomaly_score <= 1.0


def test_forensic_engine_nonexistent_file(tmp_path):
    p = tmp_path / "missing.png"
    engine = ForensicEngine(fft_size=64)
    report = engine.analyze(p)

    assert report.metadata is not None
    assert report.metadata.is_corrupted is True
    # Should not crash; errors should be recorded.
    assert len(report.errors) > 0


def test_forensic_report_as_dict(tmp_path):
    p = tmp_path / "d.png"
    _save_image(p)

    engine = ForensicEngine(fft_size=64)
    report = engine.analyze(p)
    d = report.as_dict()

    assert set(d.keys()) == {
        "path", "file_name", "directory", "extension", "size_bytes",
        "metadata", "frequency", "noise",
        "metadata_suspicious", "frequency_anomaly_score",
        "noise_anomaly_score", "composite_score", "errors",
    }
    assert isinstance(d["errors"], list)
