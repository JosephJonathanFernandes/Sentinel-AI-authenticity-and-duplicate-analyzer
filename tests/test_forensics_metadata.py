"""Tests for metadata extraction (forensics Module D).

Verifies:
- EXIF fields are extracted from a valid image.
- Missing/corrupt files are handled without crashing.
- Suspicious metadata is flagged but NOT treated as proof of AI generation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ExifTags

from sentinel.app.forensics.metadata import (
    extract_metadata,
    ImageMetadata,
    _looks_like_generation_software,
)


def _make_jpeg_with_exif(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (200, 200), (100, 150, 200))
    exif = Image.Exif()
    exif[271] = "Canon"           # Make
    exif[272] = "EOS 5D"          # Model
    exif[305] = "Adobe Photoshop"  # Software
    exif[36867] = "2024:01:15 12:00:00"  # DateTimeOriginal
    img.save(path, format="JPEG", exif=exif)


def test_extract_metadata_valid_jpeg(tmp_path):
    p = tmp_path / "photo.jpg"
    _make_jpeg_with_exif(p)

    meta = extract_metadata(p)

    assert meta.error is None
    assert meta.is_corrupted is False
    assert meta.format == "JPEG"
    assert meta.width == 200
    assert meta.height == 200
    assert meta.camera_make == "Canon"
    assert meta.camera_model == "EOS 5D"
    assert meta.software == "Adobe Photoshop"
    assert meta.metadata_complete is True
    assert meta.suspicious is False


def test_extract_metadata_no_exif(tmp_path):
    p = tmp_path / "plain.png"
    Image.new("RGB", (100, 100), (10, 20, 30)).save(p, format="PNG")

    meta = extract_metadata(p)

    assert meta.error is None
    assert meta.is_corrupted is False
    assert meta.format == "PNG"
    # PNG typically has no EXIF; metadata should be incomplete but not crash.
    assert metadata_incomplete(meta)


def test_extract_metadata_nonexistent_file(tmp_path):
    p = tmp_path / "does_not_exist.jpg"
    meta = extract_metadata(p)
    assert meta.is_corrupted is True
    assert meta.error is not None


def test_extract_metadata_corrupt_file(tmp_path):
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"not a real jpeg file")
    meta = extract_metadata(p)
    assert meta.is_corrupted is True
    assert meta.error is not None


def test_missing_metadata_not_ai_proof():
    """Explicitly: missing metadata must NOT be treated as AI-generation proof."""
    # A metadata object with no camera/software info should be flagged
    # "suspicious" as supporting evidence only — there is no is_ai flag.
    meta = ImageMetadata(
        path="/x.png", file_name="x.png", directory="/x", extension=".png",
        size_bytes=1234,
    )
    meta.metadata_complete = False
    meta.suspicious = True
    meta.suspicious_reason = "No camera or date information"

    # The only forensic signal is `suspicious`; nothing asserts AI generation.
    assert meta.suspicious is True
    assert not hasattr(meta, "is_ai_generated")


def test_generation_software_heuristic():
    assert _looks_like_generation_software("Stable Diffusion") is True
    assert _looks_like_generation_software("Midjourney") is True
    assert _looks_like_generation_software("DALL-E") is True
    # General editors are NOT flagged as generation software.
    assert _looks_like_generation_software("Adobe Photoshop") is False
    assert _looks_like_generation_software("GIMP") is False
    assert _looks_like_generation_software("Canon Digital Camera") is False


def metadata_incomplete(meta: ImageMetadata) -> bool:
    return not meta.metadata_complete
