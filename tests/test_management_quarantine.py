"""Tests for quarantine management: list, restore, delete-from-quarantine."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.app.management.quarantine import QuarantineManager, QuarantinedFile


@pytest.fixture
def qm(tmp_path):
    qdir = tmp_path / "quarantine"
    return QuarantineManager(quarantine_dir=qdir)


def test_list_quarantined_empty(qm):
    files = qm.list_quarantined()
    assert files == []


def test_quarantine_and_restore(tmp_path, qm):
    # Create a file to quarantine
    src = tmp_path / "source.png"
    src.write_bytes(b"fake image data")

    result = qm.fm.quarantine(src)
    assert result.success is True
    assert result.destination_path is not None

    quarantined = qm.list_quarantined()
    assert len(quarantined) == 1
    assert isinstance(quarantined[0], QuarantinedFile)
    assert quarantined[0].name == "source.png"

    # Restore it
    restore_dir = tmp_path / "restored"
    restore_dir.mkdir()
    res = qm.restore(quarantined[0].path, destination_dir=restore_dir)
    assert res.success is True
    assert Path(res.destination_path).exists()
    # Quarantine should now be empty
    assert qm.list_quarantined() == []


def test_delete_from_quarantine(tmp_path, qm):
    src = tmp_path / "todelete.png"
    src.write_bytes(b"data")
    qm.fm.quarantine(src)
    assert len(qm.list_quarantined()) == 1

    qp = qm.list_quarantined()[0].path
    res = qm.delete_from_quarantine(qp)
    assert res.success is True
    assert not qp.exists()
    assert qm.list_quarantined() == []


def test_quarantine_size_bytes(qm, tmp_path):
    assert qm.quarantine_size_bytes() == 0

    src = tmp_path / "qsize.png"
    src.write_bytes(b"x" * 1000)
    qm.fm.quarantine(src)
    assert qm.quarantine_size_bytes() == 1000
