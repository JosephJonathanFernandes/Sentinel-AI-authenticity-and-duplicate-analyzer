"""Near-duplicate similarity engine.

Combines multiple independent signals into a single similarity score:

1. Perceptual hashing (pHash / dHash / aHash) via Hamming distance.
2. ORB keypoint feature matching.

The final score is a weighted combination. No single algorithm is trusted in
isolation. Results are persisted as `near_duplicate_edges`.

Design decision: duplicate detection (this engine) and AI detection are kept
completely independent. A near-duplicate pair is never interpreted as evidence
about whether an image is AI-generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from ..config import settings
from ..database import models
from ..database import repository as repo
from .perceptual_hash import compute_hashes, hamming_similarity, PerceptualHashes
from .feature_matching import compute_orb, match_orb, OrbResult


@dataclass
class PairSimilarity:
    image_a: models.Image
    image_b: models.Image
    phash_similarity: Optional[float] = None
    dhash_similarity: Optional[float] = None
    ahash_similarity: Optional[float] = None
    orb_score: float = 0.0
    orb_matches: int = 0
    combined_score: float = 0.0
    method: str = "combined"
    hamming_distance: Optional[int] = None


class SimilarityEngine:
    """Finds near-duplicate image pairs using combined perceptual + ORB signals."""

    def __init__(self, session: Session):
        self.session = session

    def _hash_cache(self) -> dict[int, PerceptualHashes]:
        """Load (or compute) perceptual hashes for all non-corrupted images.

        Cached on the Image row (phash / dhash / ahash columns) to avoid
        recomputation across scans.
        """
        images = self.session.query(models.Image).filter(
            models.Image.is_corrupted == False  # noqa: E712
        ).all()
        cache: dict[int, PerceptualHashes] = {}
        for image in images:
            ph = image.phash
            dh = image.dhash
            ah = image.ahash
            if ph and dh and ah:
                cache[image.id] = PerceptualHashes(phash=ph, dhash=dh, ahash=ah)
            else:
                hashes = compute_hashes(
                    Path(image.path),
                    hash_size=settings.hash_size,
                    highhash_size=settings.highhash_size,
                )
                repo.update_image(
                    self.session,
                    image,
                    {
                        "phash": hashes.phash,
                        "dhash": hashes.dhash,
                        "ahash": hashes.ahash,
                    },
                )
                self.session.flush()
                cache[image.id] = hashes
        self.session.commit()
        return cache

    def _orb_cache(self) -> dict[int, OrbResult]:
        """Compute (or reuse) ORB descriptors. Not persisted to DB (descriptors
        are large); recomputed per comparison run to keep the DB lean."""
        images = self.session.query(models.Image).filter(
            models.Image.is_corrupted == False  # noqa: E712
        ).all()
        cache: dict[int, OrbResult] = {}
        for image in images:
            cache[image.id] = compute_orb(
                Path(image.path), max_features=settings.orb_max_features
            )
        return cache

    def compare_pair(
        self,
        image_a: models.Image,
        image_b: models.Image,
        hash_cache: dict[int, PerceptualHashes],
        orb_cache: dict[int, OrbResult],
    ) -> PairSimilarity:
        """Compute the combined similarity between two images."""
        ha = hash_cache.get(image_a.id)
        hb = hash_cache.get(image_b.id)

        phash_sim = hash_sim = None
        dhash_sim = None
        ahash_sim = None
        hamming_dist = None

        if ha and hb:
            phash_sim = hamming_similarity(ha.phash or "", hb.phash or "", settings.hash_size)
            dhash_sim = hamming_similarity(ha.dhash or "", hb.dhash or "", settings.hash_size)
            ahash_sim = hamming_similarity(ha.ahash or "", hb.ahash or "", settings.hash_size)
            hamming_dist = hamming_distance(ha.phash or "", hb.phash or "")

        oa = orb_cache.get(image_a.id)
        ob = orb_cache.get(image_b.id)
        orb_matches, orb_score = 0, 0.0
        if oa and ob and oa.descriptors is not None and ob.descriptors is not None:
            orb_matches, orb_score = match_orb(
                oa.descriptors, ob.descriptors, ratio=settings.orb_match_ratio
            )

        # Weighted combination: perceptual hash dominates; ORB adds confirmation.
        components = []
        weights = []
        if phash_sim is not None:
            components.append(phash_sim)
            weights.append(0.4)
        if dhash_sim is not None:
            components.append(dhash_sim)
            weights.append(0.2)
        if ahash_sim is not None:
            components.append(ahash_sim)
            weights.append(0.2)
        if orb_score > 0:
            components.append(orb_score)
            weights.append(0.2)

        if components:
            total_w = sum(weights)
            combined = sum(c * w for c, w in zip(components, weights)) / total_w
        else:
            combined = 0.0

        # Determine primary method
        if orb_score >= 0.5 and (phash_sim or 0) > 0:
            method = "phash+orb"
        elif phash_sim is not None and phash_sim > 0:
            method = "phash"
        elif orb_score > 0:
            method = "orb"
        else:
            method = "none"

        return PairSimilarity(
            image_a=image_a,
            image_b=image_b,
            phash_similarity=phash_sim,
            dhash_similarity=dhash_sim,
            ahash_similarity=ahash_sim,
            orb_score=orb_score,
            orb_matches=orb_matches,
            combined_score=float(combined),
            method=method,
            hamming_distance=hamming_dist,
        )

    def find_near_duplicates(
        self,
        threshold: Optional[float] = None,
        max_pairs: Optional[int] = None,
    ) -> list[models.NearDuplicateEdge]:
        """Compare candidate image pairs and persist edges above threshold.

        To keep comparisons O(n) manageable, candidate pairs are pre-filtered
        by pHash: only pairs whose pHash Hamming distance is within the
        threshold-equivalent range proceed to full comparison (including ORB).
        """
        threshold = threshold if threshold is not None else settings.similarity_combined_threshold

        # Persist fresh edges (clear previous run).
        self.session.query(models.NearDuplicateEdge).delete()
        self.session.flush()

        hash_cache = self._hash_cache()
        orb_cache = self._orb_cache()

        images = list(hash_cache.keys())
        persisted: list[models.NearDuplicateEdge] = []

        # Candidate prefilter: compare every pair on pHash first (cheap).
        # For small-to-medium collections this is fine; for very large
        # collections a FAISS index could replace this O(n^2) step.
        for i in range(len(images)):
            id_a = images[i]
            img_a = self.session.get(models.Image, id_a)
            ha = hash_cache[id_a]
            for j in range(i + 1, len(images)):
                id_b = images[j]
                img_b = self.session.get(models.Image, id_b)
                hb = hash_cache[id_b]

                # Cheap pHash prefilter
                if ha.phash and hb.phash:
                    dist = hamming_distance(ha.phash, hb.phash)
                    # Rough prefilter: allow pairs within a generous Hamming radius.
                    if dist is not None and dist > settings.hash_size * settings.hash_size:
                        continue

                pair = self.compare_pair(img_a, img_b, hash_cache, orb_cache)
                if pair.combined_score >= threshold:
                    edge = repo.add_near_duplicate_edge(
                        self.session,
                        image_a=img_a,
                        image_b=img_b,
                        similarity_score=pair.combined_score,
                        hamming_distance=pair.hamming_distance,
                        method=pair.method,
                    )
                    persisted.append(edge)
                    if max_pairs and len(persisted) >= max_pairs:
                        self.session.commit()
                        return persisted

        self.session.commit()
        return persisted


def hamming_distance(hash_a: str, hash_b: str) -> Optional[int]:
    from .perceptual_hash import hamming_distance as _hd
    return _hd(hash_a, hash_b)
