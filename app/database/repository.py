"""Repository layer: CRUD and query helpers for the Sentinel domain models.

All database access funnels through this module so the rest of the app
remains independent of SQLAlchemy details.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from . import models


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def get_image_by_path(session: Session, path: str) -> Optional[models.Image]:
    return session.execute(
        select(models.Image).where(models.Image.path == path)
    ).scalar_one_or_none()


def create_image(session: Session, image_data: dict) -> models.Image:
    image = models.Image(**image_data)
    session.add(image)
    session.flush()
    return image


def update_image(session: Session, image: models.Image, updates: dict) -> models.Image:
    for key, value in updates.items():
        setattr(image, key, value)
    session.add(image)
    session.flush()
    return image


def list_images(session: Session, limit: int = 1000, offset: int = 0):
    return session.execute(
        select(models.Image).offset(offset).limit(limit)
    ).scalars().all()


def count_images(session: Session) -> int:
    return session.execute(select(func.count(models.Image.id))).scalar() or 0


def count_by_extension(session: Session):
    rows = session.execute(
        select(models.Image.extension, func.count(models.Image.id))
        .group_by(models.Image.extension)
    ).all()
    return {ext: cnt for ext, cnt in rows}


def delete_image(session: Session, image: models.Image) -> None:
    session.delete(image)
    session.flush()


# ---------------------------------------------------------------------------
# Exact duplicates
# ---------------------------------------------------------------------------
def get_exact_groups(session: Session):
    return session.execute(
        select(models.ExactDuplicateGroup)
        .options(joinedload(models.ExactDuplicateGroup.members).joinedload(models.ExactDuplicateMember.image))
        .order_by(models.ExactDuplicateGroup.id.desc())
    ).scalars().all()


def create_exact_group(
    session: Session,
    sha256: str,
    file_count: int,
    total_size_bytes: int,
    redundant_size_bytes: int,
) -> models.ExactDuplicateGroup:
    group = models.ExactDuplicateGroup(
        sha256=sha256,
        file_count=file_count,
        total_size_bytes=total_size_bytes,
        redundant_size_bytes=redundant_size_bytes,
    )
    session.add(group)
    session.flush()
    return group


def add_exact_member(
    session: Session,
    group: models.ExactDuplicateGroup,
    image: models.Image,
    is_primary: bool = False,
) -> models.ExactDuplicateMember:
    member = models.ExactDuplicateMember(
        group_id=group.id,
        image_id=image.id,
        is_primary=is_primary,
    )
    session.add(member)
    session.flush()
    return member


# ---------------------------------------------------------------------------
# Near duplicates
# ---------------------------------------------------------------------------
def add_near_duplicate_edge(
    session: Session,
    image_a: models.Image,
    image_b: models.Image,
    similarity_score: float,
    hamming_distance: Optional[int] = None,
    method: str = "combined",
) -> models.NearDuplicateEdge:
    edge = models.NearDuplicateEdge(
        image_a_id=image_a.id,
        image_b_id=image_b.id,
        similarity_score=similarity_score,
        hamming_distance=hamming_distance,
        method=method,
    )
    session.add(edge)
    session.flush()
    return edge


def get_near_duplicate_edges(session: Session, limit: int = 500):
    return session.execute(
        select(models.NearDuplicateEdge)
        .options(
            joinedload(models.NearDuplicateEdge.image_a),
            joinedload(models.NearDuplicateEdge.image_b),
        )
        .order_by(models.NearDuplicateEdge.similarity_score.desc())
        .limit(limit)
    ).scalars().all()


# ---------------------------------------------------------------------------
# AI detection
# ---------------------------------------------------------------------------
def set_ai_result(
    session: Session,
    image: models.Image,
    label: str,
    confidence: float,
    model_version: Optional[str] = None,
    features_used: Optional[str] = None,
    explanation: Optional[str] = None,
) -> models.AIDetectionResult:
    result = models.AIDetectionResult(
        image_id=image.id,
        label=label,
        confidence=confidence,
        model_version=model_version,
        features_used=features_used,
        explanation=explanation,
    )
    session.add(result)
    session.flush()
    return result


def get_ai_results(session: Session, limit: int = 1000):
    return session.execute(
        select(models.AIDetectionResult)
        .options(joinedload(models.AIDetectionResult.image))
        .order_by(models.AIDetectionResult.created_at.desc())
        .limit(limit)
    ).scalars().all()


# ---------------------------------------------------------------------------
# Forensics
# ---------------------------------------------------------------------------
def set_forensic_result(
    session: Session,
    image: models.Image,
    fft_energy_ratio: Optional[float] = None,
    dct_energy_ratio: Optional[float] = None,
    noise_residual_std: Optional[float] = None,
    compression_estimate: Optional[str] = None,
    feature_summary: Optional[str] = None,
) -> models.ForensicResult:
    result = models.ForensicResult(
        image_id=image.id,
        fft_energy_ratio=fft_energy_ratio,
        dct_energy_ratio=dct_energy_ratio,
        noise_residual_std=noise_residual_std,
        compression_estimate=compression_estimate,
        feature_summary=feature_summary,
    )
    session.add(result)
    session.flush()
    return result


# ---------------------------------------------------------------------------
# Scan history / user actions
# ---------------------------------------------------------------------------
def create_scan_history(session: Session, scan_data: dict) -> models.ScanHistory:
    history = models.ScanHistory(**scan_data)
    session.add(history)
    session.flush()
    return history


def update_scan_history(session: Session, history: models.ScanHistory, updates: dict) -> models.ScanHistory:
    for key, value in updates.items():
        setattr(history, key, value)
    session.add(history)
    session.flush()
    return history


def add_user_action(
    session: Session,
    action_type: str,
    source_path: Optional[str] = None,
    destination_path: Optional[str] = None,
    details: Optional[str] = None,
    image: Optional[models.Image] = None,
) -> models.UserAction:
    action = models.UserAction(
        action_type=action_type,
        image_id=image.id if image else None,
        source_path=source_path,
        destination_path=destination_path,
        details=details,
    )
    session.add(action)
    session.flush()
    return action
