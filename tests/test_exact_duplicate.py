"""Tests for exact duplicate detection (SHA-256).

Uses synthetic images in a temporary directory. Images that are byte-identical
must group together; images differing by even one byte must not.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from sentinel.app.duplicate_detection.exact_hash import (
    compute_sha256,
    ExactDuplicateEngine,
)
from sentinel.app.database.database import create_engine_for, init_db, get_session
from sentinel.app.database import models, repository as repo
from sentinel.app.core.image_loader import get_basic_info


def _make_image(path: Path, color=(120, 200, 90), size=(64, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path, format="PNG")
    return path


def _scan_and_store(session, directory: Path):
    """Populate the images table from a directory (minimal scan)."""
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".webp"}:
            info = get_basic_info(p)
            if info["is_corrupted"]:
                continue
            repo.create_image(
                session,
                {
                    "path": str(p),
                    "file_name": p.name,
                    "directory": str(p.parent),
                    "extension": p.suffix.lower(),
                    "size_bytes": p.stat().st_size,
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


def test_sha256_identical_files_same_digest(tmp_path):
    a = _make_image(tmp_path / "a" / "img.png")
    b = _make_image(tmp_path / "b" / "copy.png")
    assert compute_sha256(a) == compute_sha256(b)


def test_sha256_one_byte_changed_different_digest(tmp_path):
    a = _make_image(tmp_path / "a.png")
    b_path = tmp_path / "b.png"
    data = bytearray(a.read_bytes())
    data[-1] ^= 0xFF  # flip last byte
    b_path.write_bytes(data)
    assert compute_sha256(a) != compute_sha256(b_path)


def test_exact_duplicate_grouping(tmp_path, db_session):
    # Create duplicates across two directories plus one different image.
    dup1 = _make_image(tmp_path / "dir1" / "photo.png")
    dup2 = _make_image(tmp_path / "dir2" / "photo_copy.png")
    unique = _make_image(tmp_path / "dir1" / "other.png", color=(10, 10, 10))

    _scan_and_store(db_session, tmp_path)

    engine = ExactDuplicateEngine(db_session)
    groups = engine.build_groups(min_group_size=2)

    assert len(groups) == 1, f"Expected 1 exact group, got {len(groups)}"
    group = groups[0]
    assert group.file_count == 2
    assert group.redundant_size_bytes == group.total_size_bytes - (
        group.members[0].image.size_bytes or 0
    )
    paths = {m.image.path for m in group.members}
    assert str(dup1) in paths
    assert str(dup2) in paths


def test_exact_duplicate_no_group_for_unique_images(tmp_path, db_session):
    _make_image(tmp_path / "a.png", color=(10, 10, 10))
    _make_image(tmp_path / "b.png", color=(200, 200, 200))

    _scan_and_store(db_session, tmp_path)

    engine = ExactDuplicateEngine(db_session)
    groups = engine.build_groups(min_group_size=2)
    assert groups == []


def test_exact_duplicate_primary_is_highest_resolution(tmp_path, db_session):
    low = _make_image(tmp_path / "low.png", size=(32, 32))
    high = _make_image(tmp_path / "high.png", size=(128, 128))
    # Make them byte-identical by copying high over low
    low.write_bytes(high.read_bytes())

    _scan_and_store(db_session, tmp_path)

    engine = ExactDuplicateEngine(db_session)
    groups = engine.build_groups(min_group_size=2)
    assert len(groups) == 1
    primaries = [m for m in groups[0].members if m.is_primary]
    assert len(primaries) == 1
    assert primaries[0].image.path == str(high)
