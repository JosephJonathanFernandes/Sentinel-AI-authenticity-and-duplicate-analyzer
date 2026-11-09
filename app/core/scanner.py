"""Directory scanner for Sentinel.

Walks user-selected directories, discovers image files, and streams them
to the caller in bounded batches so the full collection is never held in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Iterable

from .image_loader import is_supported_extension, ImageLoadError


@dataclass
class ScanProgress:
    scanned: int = 0
    found: int = 0
    errors: int = 0
    current_file: str = ""
    completed: bool = False


class Scanner:
    """Recursively scan directories for image files."""

    def __init__(self, extensions: tuple[str, ...] | None = None):
        from ..config import settings
        self.extensions = set(extensions) if extensions else set(settings.image_extensions)

    def discover(self, directory: Path | str) -> Iterator[Path]:
        """Yield image file paths under `directory`, one at a time."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        for item in directory.rglob("*"):
            try:
                if item.is_file() and is_supported_extension(item):
                    yield item
            except (OSError, PermissionError):
                # Skip files we cannot stat / access.
                continue

    def discover_list(self, directory: Path | str) -> list[Path]:
        """Materialise the full list (caller must be mindful of RAM)."""
        return list(self.discover(directory))

    def batch_paths(
        self,
        directory: Path | str,
        batch_size: int | None = None,
    ) -> Iterator[list[Path]]:
        """Yield lists of image paths, each of size at most `batch_size`."""
        from ..config import settings
        batch_size = batch_size or settings.batch_size
        current: list[Path] = []
        for path in self.discover(directory):
            current.append(path)
            if len(current) >= batch_size:
                yield current
                current = []
        if current:
            yield current
