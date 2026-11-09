"""Image metadata extraction using Pillow (primary) with exiftool as optional fallback.

Extracts file-level and EXIF metadata without ever storing image bytes in the
database.  Missing or suspicious metadata is flagged as supporting forensic
evidence only — it is NOT treated as proof that an image is AI-generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

from PIL import Image, UnidentifiedImageError, ExifTags
from PIL.TiffImagePlugin import ImageFileDirectory_v2

from ..core.image_loader import ImageLoadError, load_image, close_image


@dataclass
class ImageMetadata:
    """Extracted metadata for a single image."""

    path: str
    file_name: str
    directory: str
    extension: str
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    color_mode: Optional[str] = None
    format: Optional[str] = None
    mime_type: Optional[str] = None

    # EXIF fields
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    date_time: Optional[str] = None
    date_time_original: Optional[str] = None
    date_time_digitized: Optional[str] = None
    image_description: Optional[str] = None
    orientation: Optional[str] = None
    exif_version: Optional[str] = None
    flash: Optional[str] = None
    focal_length: Optional[str] = None
    exposure_time: Optional[str] = None
    f_number: Optional[str] = None
    iso_speed: Optional[str] = None

    # Derived forensic flags
    metadata_complete: bool = False
    suspicious: bool = False
    suspicious_reason: Optional[str] = None
    raw_exif: Optional[str] = None  # JSON serialisation of raw EXIF

    # Error handling
    error: Optional[str] = None
    is_corrupted: bool = False


# EXIF tag IDs we care about, mapped to human-readable names.
_EXIF_TAG_MAP = {
    271: "camera_make",
    272: "camera_model",
    305: "software",
    306: "date_time",
    36867: "date_time_original",
    36868: "date_time_digitized",
    270: "image_description",
    274: "orientation",
    36864: "exif_version",
    37377: "flash",
    37386: "focal_length",
    33434: "exposure_time",
    33437: "f_number",
    34855: "iso_speed",
}


def _safe_str(value) -> Optional[str]:
    """Convert an EXIF value to a safe string, or None."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            return None
    return str(value)


def _extract_exif(img: Image.Image) -> dict:
    """Extract EXIF tags from a PIL image into a dict of tag_id -> value."""
    exif: dict = {}
    try:
        raw = img.getexif()
        if raw:
            for tag_id, value in raw.items():
                exif[tag_id] = value
    except Exception:
        # getexif() can fail on malformed files; return what we have.
        pass
    return exif


def extract_metadata(path: Path | str) -> ImageMetadata:
    """Extract file and EXIF metadata for an image file.

    Returns an ImageMetadata object with error/is_corrupted set if the file
    cannot be read.  Never raises for unreadable files.
    """
    path = Path(path)
    meta = ImageMetadata(
        path=str(path),
        file_name=path.name,
        directory=str(path.parent),
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size if path.exists() else 0,
    )

    if not path.exists():
        meta.error = "File not found"
        meta.is_corrupted = True
        return meta

    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        meta.error = f"Unsupported extension: {path.suffix}"
        meta.is_corrupted = True
        return meta

    try:
        loaded = load_image(path)
    except ImageLoadError as exc:
        meta.error = str(exc)
        meta.is_corrupted = True
        return meta

    try:
        img = loaded.image
        meta.width = img.width
        meta.height = img.height
        meta.color_mode = img.mode
        meta.format = img.format
        meta.mime_type = _mime_from_format(img.format)

        exif = _extract_exif(img)
        meta.raw_exif = _serialize_exif(exif)

        # Map known tags to fields.
        for tag_id, field_name in _EXIF_TAG_MAP.items():
            if tag_id in exif:
                setattr(meta, field_name, _safe_str(exif[tag_id]))

        # Heuristic: metadata completeness.
        has_camera = bool(meta.camera_make or meta.camera_model)
        has_software = bool(meta.software)
        has_date = bool(meta.date_time_original or meta.date_time_digitized or meta.date_time)
        meta.metadata_complete = has_camera and has_date

        # Suspicious flags (supporting evidence only — NOT proof of AI generation).
        reasons: list[str] = []
        if not has_camera and not has_software:
            reasons.append("No camera make/model or software information found")
        if path.suffix.lower() in {".jpg", ".jpeg"} and not has_exif(exif):
            reasons.append("JPEG file with no EXIF data")
        if has_software and _looks_like_generation_software(meta.software or ""):
            reasons.append(f"Software indicates possible generation/editing: {meta.software}")
        if not has_date:
            reasons.append("No date/time information in EXIF")

        if reasons:
            meta.suspicious = True
            meta.suspicious_reason = "; ".join(reasons)
    except Exception as exc:  # pragma: no cover - defensive
        meta.error = f"Metadata extraction failed: {exc}"
        meta.is_corrupted = True
    finally:
        close_image(loaded.image)

    return meta


def _mime_from_format(fmt: Optional[str]) -> Optional[str]:
    if not fmt:
        return None
    mapping = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "BMP": "image/bmp",
        "TIFF": "image/tiff",
        "WEBP": "image/webp",
    }
    return mapping.get(fmt.upper())


def _serialize_exif(exif: dict) -> Optional[str]:
    """JSON-serialise EXIF dict for storage.  Non-serialisable values are dropped."""
    out: dict = {}
    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        try:
            json.dumps(value)
            out[str(tag_name)] = value
        except (TypeError, ValueError):
            out[str(tag_name)] = _safe_str(value)
    return json.dumps(out) if out else None


def has_exif(exif: dict) -> bool:
    """Return True if the EXIF dict contains any meaningful tags."""
    return bool(exif)


def _looks_like_generation_software(software: str) -> bool:
    """Heuristic check for software strings associated with AI image generation.

    General photo editors (Photoshop, GIMP, etc.) are intentionally NOT flagged,
    because their presence is normal and not evidence of AI generation.
    """
    indicators = (
        "stable diffusion",
        "midjourney",
        "dall-e",
        "dalle",
        "openai",
        "novelai",
        "runway",
        "leaful.ai",
        "craiyon",
        "artbreeder",
    )
    lower = software.lower()
    return any(ind in lower for ind in indicators)
