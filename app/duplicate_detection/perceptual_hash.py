"""Perceptual hashing: pHash, dHash, aHash.

These hashes capture visual content rather than exact bytes, so resized,
compressed, cropped, brightness/contrast adjusted, or re-encoded images
produce similar hashes. Similarity is measured with Hamming distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

from ..core.image_loader import ImageLoadError, close_image, load_image


@dataclass
class PerceptualHashes:
    phash: Optional[str] = None
    dhash: Optional[str] = None
    ahash: Optional[str] = None
    phash_high: Optional[str] = None  # larger hash for finer granularity
    error: Optional[str] = None


def _image_to_gray(path: Path) -> Optional[Image.Image]:
    """Open an image and convert to grayscale 'L' mode. Returns None on failure."""
    try:
        loaded = load_image(path)
    except ImageLoadError:
        return None
    try:
        return loaded.image.convert("L")
    except Exception:
        close_image(loaded.image)
        return None
    finally:
        # loaded.image still owned by caller; we return the converted image which
        # shares pixel data. The caller (compute_hashes) closes it.
        pass


def compute_hashes(path: Path, hash_size: int = 8, highhash_size: int = 16) -> PerceptualHashes:
    """Compute pHash, dHash, aHash (and a larger pHash) for an image file."""
    result = PerceptualHashes()
    try:
        gray = _image_to_gray(path)
        if gray is None:
            result.error = "Unable to decode image"
            return result

        try:
            ph = imagehash.phash(gray, hash_size=hash_size)
            dh = imagehash.dhash(gray, hash_size=hash_size)
            ah = imagehash.average_hash(gray, hash_size=hash_size)
            ph_high = imagehash.phash(gray, hash_size=highhash_size)

            result.phash = str(ph)
            result.dhash = str(dh)
            result.ahash = str(ah)
            result.phash_high = str(ph_high)
        except Exception as exc:  # pragma: no cover - defensive
            result.error = f"Hash computation failed: {exc}"
        finally:
            gray.close()
    except Exception as exc:  # pragma: no cover - defensive
        result.error = str(exc)
    return result


def hamming_distance(hash_a: str, hash_b: str) -> Optional[int]:
    """Hamming distance between two hex-encoded imagehash strings.

    Returns None if either hash is missing or they differ in length.
    """
    if not hash_a or not hash_b:
        return None
    if len(hash_a) != len(hash_b):
        return None
    try:
        int_a = int(hash_a, 16)
        int_b = int(hash_b, 16)
    except ValueError:
        return None
    return bin(int_a ^ int_b).count("1")


def hamming_similarity(hash_a: str, hash_b: str, hash_size: int = 8) -> Optional[float]:
    """Convert Hamming distance to a 0..1 similarity score."""
    dist = hamming_distance(hash_a, hash_b)
    if dist is None:
        return None
    total_bits = hash_size * hash_size
    if total_bits == 0:
        return None
    return 1.0 - (dist / total_bits)
