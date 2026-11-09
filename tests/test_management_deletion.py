"""Tests for safe deletion: validation, quarantine, permanent delete with confirmation."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.app.management.deletion import SafeDeletionManager, DeletionRequest


@pytest.fixture
def sdm(tmp_path):
    return SafeDeletionManager(quarantine_by_default=True)


def test_validate_nonexistent_file(sdm, tmp_path):
    ok, err = sdm.validate(tmp_path / "missing.png")
    assert ok is False
    assert err is not None


def test_validate_existing_file(sdm, tmp_path):
    p = tmp_path / "exists.png"
    p.write_bytes(b"data")
    ok, err = sdm.validate(p)
    assert ok is True
    assert err is None


def test_quarantine_moves_file(sdm, tmp_path):
    src = tmp_path / "to_quarantine.png"
    src.write_bytes(b"data")

    result = sdm.quarantine(src)
    assert result.success is True
    assert not src.exists()  # moved out of original location
    assert result.destination_path is not None


def test_delete_permanent_requires_confirmation(sdm, tmp_path):
    p = tmp_path / "nodelete.png"
    p.write_bytes(b"data")

    result = sdm.delete_permanent(p, confirmed=False)
    assert result.success is False
    assert "not confirmed" in result.message
    assert p.exists()  # file still there


def test_delete_permanent_with_confirmation(sdm, tmp_path):
    p = tmp_path / "todelete.png"
    p.write_bytes(b"data")

    result = sdm.delete_permanent(p, confirmed=True)
    assert result.success is True
    assert not p.exists()


def test_deletion_request_dataclass():
    req = DeletionRequest(
        image_id=1,
        source_path="/x/y.png",
        file_name="y.png",
        reason="Duplicate of higher-res copy",
    )
    assert req.image_id == 1
    assert req.file_name == "y.png"
    assert req.reason == "Duplicate of higher-res copy"
