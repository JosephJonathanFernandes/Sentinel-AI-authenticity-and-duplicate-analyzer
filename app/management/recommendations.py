"""Image retention recommendation engine.

For each near-duplicate or exact-duplicate group, this module scores every
candidate copy and recommends the best one to keep.  The score is based on:

* Resolution (higher is better)
* File size (larger, less-compressed is better)
* Metadata completeness (more EXIF = more trustworthy)
* Image quality indicators (sharpness via Laplacian variance)

The explanation is human-readable so users understand *why* a particular
image is recommended.  No image is ever deleted automatically — the UI
presents the recommendation and the user decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ..core.image_loader import load_image, close_image
from ..database import models
from ..database import repository as repo


@dataclass
class Recommendation:
    """Recommendation for a single image within a duplicate group."""

    image: models.Image
    score: float                      # 0..100, higher = better to keep
    rank: int = 0                     # 1 = best
    reasons: list[str] = field(default_factory=list)  # human-readable why
    is_recommended: bool = False      # True for the top-ranked image


@dataclass
class GroupRecommendation:
    """Recommendations for one duplicate group."""

    group_id: Optional[int]           # None for near-dup pairs
    group_type: str                   # "exact" | "near_duplicate"
    images: list[Recommendation]
    recommended_image: Optional[models.Image] = None

    @property
    def primary(self) -> Optional[Recommendation]:
        return self.images[0] if self.images else None


class RecommendationEngine:
    """Scores and ranks images within duplicate groups."""

    # Weights for the composite score (must sum to 1.0).
    _W_RESOLUTION = 0.35
    _W_FILE_SIZE = 0.25
    _W_METADATA = 0.20
    _W_QUALITY = 0.20

    def __init__(self, session):
        self.session = session

    def recommend_for_exact_group(
        self, group: models.ExactDuplicateGroup
    ) -> GroupRecommendation:
        """Rank all members of an exact-duplicate group."""
        # members are ExactDuplicateMember objects; extract the Image.
        members = [m.image for m in group.members]
        return self._rank_images(members, group.id, "exact")

    def recommend_for_near_duplicate_pair(
        self, image_a: models.Image, image_b: models.Image
    ) -> GroupRecommendation:
        """Rank two images in a near-duplicate pair."""
        return self._rank_images([image_a, image_b], None, "near_duplicate")

    def _rank_images(
        self,
        images: list[models.Image],
        group_id: Optional[int],
        group_type: str,
    ) -> GroupRecommendation:
        """Compute scores, rank, and attach explanations."""
        scored: list[Recommendation] = []

        # Precompute maxima for normalisation.
        max_res = max(
            ((i.width or 0) * (i.height or 0)) for i in images
        ) or 1
        max_size = max(i.size_bytes or 0 for i in images) or 1

        for image in images:
            rec = self._score_single(image, max_res, max_size)
            scored.append(rec)

        # Sort descending by score.
        scored.sort(key=lambda r: r.score, reverse=True)

        for rank, rec in enumerate(scored, start=1):
            rec.rank = rank
            rec.is_recommended = (rank == 1)

        recommended = scored[0] if scored else None
        return GroupRecommendation(
            group_id=group_id,
            group_type=group_type,
            images=scored,
            recommended_image=recommended.image if recommended else None,
        )

    def _score_single(
        self,
        image: models.Image,
        max_res: int,
        max_size: int,
    ) -> Recommendation:
        """Score one image and build a human-readable explanation."""
        reasons: list[str] = []

        # ---- Resolution score (0..1) ----
        res = (image.width or 0) * (image.height or 0)
        res_score = res / max_res if max_res else 0.0
        if res_score > 0.9:
            reasons.append("Highest resolution in the group")
        elif res_score > 0.5:
            reasons.append("Good resolution")
        else:
            reasons.append("Lower resolution than other copies")

        # ---- File size score (0..1) ----
        size_score = (image.size_bytes or 0) / max_size if max_size else 0.0
        if size_score > 0.9:
            reasons.append("Largest file size (least compression)")
        elif size_score > 0.5:
            reasons.append("Moderate file size")
        else:
            reasons.append("Smaller file size (more compressed)")

        # ---- Metadata completeness score (0..1) ----
        meta = self._get_metadata(image)
        meta_score = 0.0
        if meta:
            fields_present = sum(
                1 for f in (
                    meta.camera_make, meta.camera_model, meta.software,
                    meta.date_time_original, meta.date_time,
                ) if f
            )
            meta_score = fields_present / 5.0
            if meta.metadata_complete:
                reasons.append("Complete camera metadata")
            elif fields_present >= 3:
                reasons.append("Partial camera metadata")
            else:
                reasons.append("Limited or no camera metadata")
        else:
            reasons.append("No metadata available")

        # ---- Quality score (sharpness via Laplacian variance, 0..1) ----
        quality_score = self._sharpness_score(image)
        if quality_score > 0.7:
            reasons.append("Sharp image (high detail)")
        elif quality_score > 0.4:
            reasons.append("Moderate sharpness")
        else:
            reasons.append("Softer / lower detail")

        # ---- Composite (0..100) ----
        composite = (
            self._W_RESOLUTION * res_score
            + self._W_FILE_SIZE * size_score
            + self._W_METADATA * meta_score
            + self._W_QUALITY * quality_score
        ) * 100.0

        return Recommendation(
            image=image,
            score=round(composite, 1),
            reasons=reasons,
        )

    def _get_metadata(self, image: models.Image):
        """Fetch the ImageMetadata row for an image, if present."""
        return self.session.query(models.ImageMetadata).filter(
            models.ImageMetadata.image_id == image.id
        ).first()

    def _sharpness_score(self, image: models.Image) -> float:
        """Estimate sharpness using Laplacian variance (0..1, higher = sharper).

        Loads the image, converts to grayscale, computes Laplacian, and
        normalises against a reference constant.  Memory-bounded: the image
        is resized to a max dimension before analysis.
        """
        try:
            loaded = load_image(Path(image.path))
        except Exception:
            return 0.0

        try:
            img = loaded.image.convert("L")
            # Resize for bounded computation.
            max_dim = 512
            if max(img.width, img.height) > max_dim:
                img.thumbnail((max_dim, max_dim))
            arr = np.asarray(img, dtype=np.float32)
            laplacian_var = float(cv2.Laplacian(arr, cv2.CV_32F).var())
            # Normalise: typical sharp photos are ~100-1000; soft images < 50.
            score = min(1.0, laplacian_var / 300.0)
            return score
        except Exception:
            return 0.0
        finally:
            close_image(loaded.image)
