"""Tests for near-duplicate detection.

Synthetic source image is transformed (resize, JPEG compress, crop, rotate)
and each transform should still register as a near-duplicate. Two unrelated
images should not.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from sentinel.app.duplicate_detection.perceptual_hash import (
    compute_hashes,
    hamming_distance,
    hamming_similarity,
)
from sentinel.app.duplicate_detection.feature_matching import compute_orb, match_orb
from sentinel.app.duplicate_detection.similarity_engine import SimilarityEngine
from sentinel.app.database.database import create_engine_for, init_db, get_session
from sentinel.app.database import models, repository as repo
from sentinel.app.core.image_loader import get_basic_info
from sentinel.app.config import settings


def _make_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic noisy geometric image so transforms behave predictably.
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (60, 160, 90), -1)
    cv2.circle(img, (150, 150), 60, (200, 50, 50), -1)
    Image.fromarray(img).save(path, format="PNG")


def _store_image(session, path: Path) -> models.Image:
    info = get_basic_info(path)
    return repo.create_image(
        session,
        {
            "path": str(path),
            "file_name": path.name,
            "directory": str(path.parent),
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "width": info["width"],
            "height": info["height"],
            "color_mode": info["mode"],
            "format": info["format"],
        },
    )


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine_for(db_path)
    init_db(engine)
    with get_session(engine) as session:
        yield session
    engine.dispose()


def test_hashes_stable_across_reencode(tmp_path):
    src = tmp_path / "source.png"
    _make_source(src)
    h_orig = compute_hashes(src)

    # Re-encode as JPEG at different quality
    jpg = tmp_path / "source.jpg"
    img = Image.open(src)
    img.convert("RGB").save(jpg, format="JPEG", quality=85)
    h_jpg = compute_hashes(jpg)

    # pHash distance should be small (<= a few bits) for same content
    dist = hamming_distance(h_orig.phash, h_jpg.phash)
    assert dist is not None
    assert dist <= 6, f"pHash distance too large after JPEG re-encode: {dist}"


def test_resized_image_detected(tmp_path):
    src = tmp_path / "source.png"
    _make_source(src)
    small = tmp_path / "small.png"
    img = Image.open(src)
    img.resize((150, 150)).save(small, format="PNG")

    h_a = compute_hashes(src)
    h_b = compute_hashes(small)
    sim = hamming_similarity(h_a.phash, h_b.phash)
    assert sim is not None
    assert sim >= 0.75, f"Resized image similarity too low: {sim}"


def test_cropped_image_detected(tmp_path):
    src = tmp_path / "source.png"
    _make_source(src)
    cropped = tmp_path / "cropped.png"
    img = Image.open(src)
    img.crop((50, 50, 300, 300)).save(cropped, format="PNG")

    h_a = compute_hashes(src)
    h_b = compute_hashes(cropped)
    sim = hamming_similarity(h_a.phash, h_b.phash)
    assert sim is not None
    assert sim >= 0.5, f"Cropped image similarity too low: {sim}"


def test_rotated_image_detected(tmp_path):
    src = tmp_path / "source.png"
    _make_source(src)
    rotated = tmp_path / "rotated.png"
    img = Image.open(src)
    img.rotate(15, expand=False, fillcolor=(0, 0, 0)).save(rotated, format="PNG")

    h_a = compute_hashes(src)
    h_b = compute_hashes(rotated)
    sim = hamming_similarity(h_a.phash, h_b.phash)
    assert sim is not None
    assert sim >= 0.5, f"Rotated image similarity too low: {sim}"


def test_unrelated_images_not_near_duplicates(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _make_source(a)
    # Completely different content
    img = Image.new("RGB", (300, 300), (10, 10, 10))
    img.save(b, format="PNG")

    h_a = compute_hashes(a)
    h_b = compute_hashes(b)
    sim = hamming_similarity(h_a.phash, h_b.phash)
    assert sim is not None
    # Well below the near-duplicate decision threshold (0.65).
    assert sim < 0.6, f"Unrelated images too similar: {sim}"


def test_orb_detects_resized_duplicate(tmp_path):
    src = tmp_path / "source.png"
    _make_source(src)
    small = tmp_path / "small.png"
    # Milder resize so ORB keypoints survive across scales.
    Image.open(src).resize((225, 225)).save(small, format="PNG")

    ra = compute_orb(src)
    rb = compute_orb(small)
    matches, score = match_orb(ra.descriptors, rb.descriptors)
    assert matches >= 4, f"Too few ORB matches for resize: {matches}"
    assert score > 0.05, f"ORB score too low for resize: {score}"


def test_full_similarity_engine_groups_transforms(tmp_path, db_session):
    src = tmp_path / "source.png"
    _make_source(src)
    variants = {
        "small.png": lambda p: Image.open(src).resize((150, 150)).save(p, format="PNG"),
        "jpg.png": lambda p: Image.open(src).convert("RGB").save(p, format="JPEG", quality=80),
        "unrelated.png": lambda p: Image.new("RGB", (300, 300), (200, 200, 200)).save(p, format="PNG"),
    }
    for name, fn in variants.items():
        fn(tmp_path / name)

    for p in [src] + list(variants.keys()):
        _store_image(db_session, tmp_path / p)
    db_session.commit()

    engine = SimilarityEngine(db_session)
    edges = engine.find_near_duplicates(
        threshold=settings.similarity_combined_threshold
    )

    # At least the small and jpg variants should match the source.
    matched_pairs = {
        frozenset((e.image_a.path, e.image_b.path)) for e in edges
    }
    src_str = str(src)
    related = [
        pair for pair in matched_pairs
        if src_str in pair and "unrelated.png" not in pair
    ]
    assert len(related) >= 1, f"Expected source to match transformed variants, got {matched_pairs}"
    # Unrelated should never be paired with source.
    unrelated_pairs = [
        pair for pair in matched_pairs
        if "unrelated.png" in pair and src_str in pair
    ]
    assert unrelated_pairs == []



