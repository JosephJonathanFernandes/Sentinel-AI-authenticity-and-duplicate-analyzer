"""Safe file operations for image management.

Guiding principle: never delete permanently by default. Move files into a
quarantine directory and log every destructive action.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil
import uuid
from datetime import datetime

from ..config import QUARANTINE_DIR, settings


@dataclass
class FileOperationResult:
    success: bool
    action: str            # "quarantine" | "delete" | "restore"
    source_path: str
    destination_path: Optional[str]
    message: str


class FileManager:
    """Manage safe relocation / deletion of images."""

    def __init__(self, quarantine_dir: Path | str | None = None):
        self.quarantine_dir = Path(quarantine_dir) if quarantine_dir else QUARANTINE_DIR
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _unique_destination(self, dest_dir: Path, source_path: Path) -> Path:
        """Return a non-colliding destination path inside dest_dir."""
        dest = dest_dir / source_path.name
        if not dest.exists():
            return dest
        stem = source_path.stem
        suffix = source_path.suffix
        counter = 1
        while True:
            candidate = dest_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def quarantine(self, source_path: Path | str) -> FileOperationResult:
        """Move a file into the quarantine folder. Default safe strategy."""
        source = Path(source_path)
        if not source.exists():
            return FileOperationResult(
                success=False, action="quarantine", source_path=str(source),
                destination_path=None, message=f"Source not found: {source}",
            )
        try:
            destination = self._unique_destination(self.quarantine_dir, source)
            shutil.move(str(source), str(destination))
            return FileOperationResult(
                success=True, action="quarantine", source_path=str(source),
                destination_path=str(destination),
                message=f"Quarantined to {destination}",
            )
        except (OSError, PermissionError, shutil.Error) as exc:
            return FileOperationResult(
                success=False, action="quarantine", source_path=str(source),
                destination_path=None, message=str(exc),
            )

    def delete_permanent(self, source_path: Path | str) -> FileOperationResult:
        """Permanently delete a file. Only when the user explicitly chooses it."""
        source = Path(source_path)
        if not source.exists():
            return FileOperationResult(
                success=False, action="delete", source_path=str(source),
                destination_path=None, message=f"Source not found: {source}",
            )
        try:
            source.unlink()
            return FileOperationResult(
                success=True, action="delete", source_path=str(source),
                destination_path=None, message="Permanently deleted",
            )
        except (OSError, PermissionError) as exc:
            return FileOperationResult(
                success=False, action="delete", source_path=str(source),
                destination_path=None, message=str(exc),
            )

    def restore(self, quarantined_path: Path | str, destination_dir: Path | str | None = None) -> FileOperationResult:
        """Restore a quarantined file to its original name / a new directory."""
        source = Path(quarantined_path)
        if not source.exists():
            return FileOperationResult(
                success=False, action="restore", source_path=str(source),
                destination_path=None, message=f"Quarantined file not found: {source}",
            )
        dest_dir = Path(destination_dir) if destination_dir else self.quarantine_dir.parent
        try:
            destination = self._unique_destination(dest_dir, source)
            shutil.move(str(source), str(destination))
            return FileOperationResult(
                success=True, action="restore", source_path=str(source),
                destination_path=str(destination), message=f"Restored to {destination}",
            )
        except (OSError, PermissionError, shutil.Error) as exc:
            return FileOperationResult(
                success=False, action="restore", source_path=str(source),
                destination_path=None, message=str(exc),
            )

    def list_quarantined(self) -> list[Path]:
        """List files currently in quarantine."""
        if not self.quarantine_dir.exists():
            return []
        return sorted([p for p in self.quarantine_dir.iterdir() if p.is_file()])

    def quarantine_size_bytes(self) -> int:
        total = 0
        for p in self.list_quarantined():
            try:
                total += p.stat().st_size
            except OSError:
                continue
        return total
