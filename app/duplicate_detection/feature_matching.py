"""ORB keypoint detection and feature matching.

ORB (Oriented FAST and Rotated BRIEF) is a free, CPU-friendly alternative to
SIFT/SURF. It is robust to rotation, scale changes, and moderate affine
transforms, making it a useful independent signal for near-duplicate detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..core.image_loader import ImageLoadError, close_image, load_image


@dataclass
class OrbResult:
    keypoints: int = 0
    descriptors: Optional[np.ndarray] = None
    error: Optional[str] = None


def compute_orb(path: Path, max_features: int = 500) -> OrbResult:
    """Detect ORB keypoints and compute descriptors for an image."""
    result = OrbResult()
    try:
        loaded = load_image(path)
    except ImageLoadError as exc:
        result.error = str(exc)
        return result

    try:
        gray = loaded.image.convert("L") if loaded.image.mode != "L" else loaded.image
        gray_np = np.array(gray)

        orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8)
        keypoints, descriptors = orb.detectAndCompute(gray_np, None)

        result.keypoints = len(keypoints) if keypoints else 0
        result.descriptors = descriptors
    except Exception as exc:  # pragma: no cover - defensive
        result.error = str(exc)
    finally:
        close_image(loaded.image)
    return result


def match_orb(
    desc_a: Optional[np.ndarray],
    desc_b: Optional[np.ndarray],
    ratio: float = 0.75,
) -> tuple[int, float]:
    """Match two ORB descriptor sets using BFMatcher + Lowe's ratio test.

    Returns (good_match_count, match_score) where match_score is the ratio of
    good matches to the smaller descriptor set (0..1).
    """
    if desc_a is None or desc_b is None:
        return 0, 0.0
    if len(desc_a) == 0 or len(desc_b) == 0:
        return 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return 0, 0.0

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    base = min(len(desc_a), len(desc_b))
    score = len(good) / base if base > 0 else 0.0
    return len(good), float(score)
