"""Safe deletion with explicit confirmation workflow.

No file is ever deleted without the caller explicitly requesting it.  This
module wraps FileManager and adds pre-deletion validation plus audit logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.file_manager import FileManager, FileOperationResult
from ..database.database import get_session, create_engine_for
from ..database import repository as repo


@dataclass
class DeletionRequest:
    """A pending deletion awaiting confirmation."""

    image_id: Optional[int]
    source_path: str
    file_name: str
    reason: str = "Duplicate management"


class SafeDeletionManager:
    """Coordinate safe deletion: validate, confirm, execute, log."""

    def __init__(self, quarantine_by_default: bool = True):
        self.fm = FileManager()
        self.quarantine_by_default = quarantine_by_default

    def validate(self, path: Path | str) -> tuple[bool, Optional[str]]:
        """Check whether a file can be deleted.  Returns (ok, error_msg)."""
        p = Path(path)
        if not p.exists():
            return False, f"File not found: {p}"
        if not p.is_file():
            return False, f"Not a file: {p}"
        try:
            p.stat()
        except (OSError, PermissionError) as exc:
            return False, f"Cannot access file: {exc}"
        return True, None

    def quarantine(self, path: Path | str, image=None) -> FileOperationResult:
        """Move a file to quarantine (default safe strategy)."""
        result = self.fm.quarantine(path)
        if result.success:
            self._log("quarantine", path=result.source_path, destination=result.destination_path, image=image)
        return result

    def delete_permanent(self, path: Path | str, image=None, confirmed: bool = False) -> FileOperationResult:
        """Permanently delete a file only if `confirmed` is True."""
        if not confirmed:
            return FileOperationResult(
                success=False, action="delete",
                source_path=str(path), destination_path=None,
                message="Deletion not confirmed — set confirmed=True to proceed",
            )
        ok, err = self.validate(path)
        if not ok:
            return FileOperationResult(
                success=False, action="delete",
                source_path=str(path), destination_path=None,
                message=err or "Validation failed",
            )
        result = self.fm.delete_permanent(path)
        if result.success:
            self._log("delete", path=result.source_path, image=image)
        return result

    def _log(
        self,
        action_type: str,
        path: Optional[str] = None,
        destination: Optional[str] = None,
        image=None,
    ) -> None:
        # Audit logging is best-effort: it must never break the primary
        # quarantine/delete operation if the DB is unavailable.
        try:
            engine = create_engine_for()
            with get_session(engine) as session:
                repo.add_user_action(
                    session,
                    action_type=action_type,
                    image=image,
                    source_path=path,
                    destination_path=destination,
                    details=f"{action_type} via management module",
                )
                session.commit()
        except Exception:
            pass
