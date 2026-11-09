"""Quarantine management: list, restore, and inspect quarantined files.

Wraps :class:`sentinel.app.core.file_manager.FileManager` and adds
database audit integration so every restore/delete is logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.file_manager import FileManager, FileOperationResult
from ..database.database import get_session, create_engine_for
from ..database import repository as repo


@dataclass
class QuarantinedFile:
    """A file currently in quarantine."""

    path: Path
    name: str
    size_bytes: int
    original_path: Optional[str] = None  # best-effort, from filename


class QuarantineManager:
    """Manage the quarantine folder and audit log."""

    def __init__(self, quarantine_dir=None):
        self.fm = FileManager(quarantine_dir)

    def list_quarantined(self) -> list[QuarantinedFile]:
        """Return all files currently in quarantine."""
        result: list[QuarantinedFile] = []
        for p in self.fm.list_quarantined():
            result.append(
                QuarantinedFile(
                    path=p,
                    name=p.name,
                    size_bytes=p.stat().st_size if p.exists() else 0,
                    original_path=None,
                )
            )
        return result

    def restore(self, quarantined_path: Path | str, destination_dir=None) -> FileOperationResult:
        """Restore a quarantined file and log the action."""
        qp = Path(quarantined_path)
        result = self.fm.restore(qp, destination_dir)
        if result.success:
            self._log_action("restore", source_path=str(qp), destination_path=result.destination_path)
        return result

    def delete_from_quarantine(self, quarantined_path: Path | str) -> FileOperationResult:
        """Permanently delete a file from quarantine and log the action."""
        qp = Path(quarantined_path)
        # FileManager doesn't expose delete-from-quarantine; do it directly.
        try:
            qp.unlink()
            result = FileOperationResult(
                success=True, action="delete_from_quarantine",
                source_path=str(qp), destination_path=None,
                message="Deleted from quarantine",
            )
            self._log_action("delete_from_quarantine", source_path=str(qp))
        except (OSError, PermissionError) as exc:
            result = FileOperationResult(
                success=False, action="delete_from_quarantine",
                source_path=str(qp), destination_path=None,
                message=str(exc),
            )
        return result

    def quarantine_size_bytes(self) -> int:
        return self.fm.quarantine_size_bytes()

    def _log_action(
        self,
        action_type: str,
        source_path: Optional[str] = None,
        destination_path: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        # Audit logging is best-effort: it must never break the primary
        # quarantine/restore operation if the DB is unavailable.
        try:
            engine = create_engine_for()
            with get_session(engine) as session:
                repo.add_user_action(
                    session,
                    action_type=action_type,
                    source_path=source_path,
                    destination_path=destination_path,
                    details=details,
                )
                session.commit()
        except Exception:
            pass
