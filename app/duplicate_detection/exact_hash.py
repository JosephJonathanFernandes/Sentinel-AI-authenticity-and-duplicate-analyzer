"""Exact duplicate detection via SHA-256 byte hashing.

Groups byte-identical images regardless of file name or location.
Results are written to the `exact_duplicate_groups` / `exact_duplicate_members`
tables. Previously computed hashes are reused when a file's SHA-256 is already
present on the `images` row, avoiding repeated hashing of unchanged files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from ..config import settings
from ..database import models
from ..database import repository as repo


CHUNK_SIZE = 1024 * 1024  # 1 MiB read chunks


def compute_sha256(path: Path) -> str:
    """Stream the SHA-256 digest of a file without loading it wholly into RAM."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


class ExactDuplicateEngine:
    """Builds exact-duplicate groups from scanned images."""

    def __init__(self, session: Session):
        self.session = session

    def hash_image(self, image: models.Image) -> str:
        """Return the SHA-256 for an image, reusing a cached value when present."""
        if image.sha256:
            return image.sha256
        digest = compute_sha256(Path(image.path))
        repo.update_image(self.session, image, {"sha256": digest})
        self.session.flush()
        return digest

    def build_groups(self, min_group_size: int = 2) -> list[models.ExactDuplicateGroup]:
        """Scan all images, group by SHA-256, persist groups, return them.

        Idempotent: existing groups for the same SHA-256 are refreshed rather
        than duplicated.
        """
        # Clear previously computed exact groups so rebuilds are clean.
        self.session.query(models.ExactDuplicateMember).delete()
        self.session.query(models.ExactDuplicateGroup).delete()
        self.session.flush()

        images = self.session.query(models.Image).filter(
            models.Image.is_corrupted == False  # noqa: E712
        ).all()

        # Group images by SHA-256, computing missing hashes on demand.
        groups_by_digest: dict[str, list[models.Image]] = {}
        for image in images:
            digest = self.hash_image(image)
            groups_by_digest.setdefault(digest, []).append(image)

        result: list[models.ExactDuplicateGroup] = []
        for digest, members in groups_by_digest.items():
            if len(members) < min_group_size:
                continue

            total_size = sum(m.size_bytes or 0 for m in members)
            # Redundant = everything except one (primary) copy.
            redundant_size = total_size - (members[0].size_bytes or 0)

            # Primary = highest resolution copy (tie-break by larger file size).
            primary = max(
                members,
                key=lambda m: ((m.width or 0) * (m.height or 0), m.size_bytes or 0),
            )

            group = repo.create_exact_group(
                self.session,
                sha256=digest,
                file_count=len(members),
                total_size_bytes=total_size,
                redundant_size_bytes=redundant_size,
            )

            for m in members:
                repo.add_exact_member(
                    self.session,
                    group=group,
                    image=m,
                    is_primary=(m.id == primary.id),
                )

            result.append(group)

        self.session.commit()
        return result
