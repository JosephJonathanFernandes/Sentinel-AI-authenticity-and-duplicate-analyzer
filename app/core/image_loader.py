"""Safe image loading utilities.

Responsibilities:
- Open an image file without crashing on corruption / unsupported formats.
- Return basic metadata (size, mode, format) and a PIL Image object.
- Never execute file contents; only decode known image formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError


@dataclass
class LoadedImage:
    """A successfully loaded image plus its metadata."""

    image: Image.Image
    path: Path
    width: int
    height: int
    mode: str
    format: Optional[str]
    size_bytes: int


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or is unsupported/corrupted."""


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_supported_extension(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_image(path: Path | str) -> LoadedImage:
    """Load an image file safely.

    Raises ImageLoadError for missing, unsupported, or corrupted files.
    The returned PIL Image is lazily decoded; callers should convert/copy
    as needed and close when done.
    """
    path = Path(path)

    if not path.exists():
        raise ImageLoadError(f"File not found: {path}")

    if not is_supported_extension(path):
        raise ImageLoadError(f"Unsupported file extension: {path.suffix}")

    try:
        size_bytes = path.stat().st_size
        img = Image.open(path)
        # Force header decode to catch truncated/corrupt files early.
        img.verify()
        # re-open after verify() because verify() leaves the file in an unusable state
        img = Image.open(path)
        img.load()

        return LoadedImage(
            image=img,
            path=path,
            width=img.width,
            height=img.height,
            mode=img.mode,
            format=img.format,
            size_bytes=size_bytes,
        )
    except UnidentifiedImageError as exc:
        raise ImageLoadError(f"Unidentified/corrupt image: {path}") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise ImageLoadError(f"Failed to load image {path}: {exc}") from exc


def close_image(img: Optional[Image.Image]) -> None:
    """Safely close a PIL image if it is open."""
    if img is not None:
        try:
            img.close()
        except Exception:
            pass


def get_basic_info(path: Path | str) -> dict:
    """Return lightweight metadata for a file without fully decoding pixels.

    Used during scanning to populate the `images` table quickly.
    """
    path = Path(path)
    info: dict = {
        "path": str(path),
        "file_name": path.name,
        "directory": str(path.parent),
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
        "is_corrupted": False,
        "error_message": None,
    }
    try:
        loaded = load_image(path)
        info["width"] = loaded.width
        info["height"] = loaded.height
        info["mode"] = loaded.mode
        info["format"] = loaded.format
        close_image(loaded.image)
    except ImageLoadError as exc:
        info["is_corrupted"] = True
        info["error_message"] = str(exc)
    return info
