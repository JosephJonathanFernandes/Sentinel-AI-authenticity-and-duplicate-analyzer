"""SQLAlchemy ORM models for Sentinel.

Design notes:
- Images are referenced by absolute path; the binary image is NEVER stored in the DB.
- Each scanned image gets one row in `images`, plus related rows in metadata,
  duplicate groups, near-duplicate edges, AI results, and forensics tables.
- Scan history and user actions provide an audit trail for destructive operations.
"""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Image(Base):
    """One row per scanned image file."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(1024), unique=True, index=True, nullable=False)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    directory: Mapped[str] = mapped_column(String(1024), nullable=False)
    extension: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    color_mode: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    format: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    phash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    dhash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    ahash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.UTC),
        onupdate=lambda: datetime.datetime.now(datetime.UTC),
        nullable=False,
    )
    scanned_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, nullable=True
    )
    is_corrupted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships (named image_metadata because "metadata" is reserved by
    # the SQLAlchemy Declarative API).
    image_metadata: Mapped[list["ImageMetadata"]] = relationship(
        "ImageMetadata", back_populates="image", cascade="all, delete-orphan"
    )
    ai_result: Mapped[Optional["AIDetectionResult"]] = relationship(
        "AIDetectionResult", back_populates="image", uselist=False,
        cascade="all, delete-orphan",
    )
    forensic: Mapped[Optional["ForensicResult"]] = relationship(
        "ForensicResult", back_populates="image", uselist=False,
        cascade="all, delete-orphan",
    )
    near_dup_edges: Mapped[list["NearDuplicateEdge"]] = relationship(
        "NearDuplicateEdge", foreign_keys="[NearDuplicateEdge.image_a_id]",
        back_populates="image_a", cascade="all, delete-orphan",
    )

    __table_args__ = ()


class ImageMetadata(Base):
    """EXIF / file metadata extracted with Pillow (and exiftool when available)."""

    __tablename__ = "image_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera_make: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    camera_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    software: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    date_time: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    date_time_original: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    date_time_digitized: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    image_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    orientation: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    exif_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    flash: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    focal_length: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    exposure_time: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    f_number: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    iso_speed: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    metadata_complete: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    suspicious: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    suspicious_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_exif: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string

    image: Mapped["Image"] = relationship("Image", back_populates="image_metadata")

    __table_args__ = ()


class ExactDuplicateGroup(Base):
    """A group of byte-identical images sharing the same SHA-256."""

    __tablename__ = "exact_duplicate_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    redundant_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC), nullable=False
    )

    members: Mapped[list["ExactDuplicateMember"]] = relationship(
        "ExactDuplicateMember", back_populates="group", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("sha256", name="uq_exact_group_sha256"),
    )


class ExactDuplicateMember(Base):
    """Membership of an image in an exact-duplicate group."""

    __tablename__ = "exact_duplicate_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(
        ForeignKey("exact_duplicate_groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    group: Mapped["ExactDuplicateGroup"] = relationship(
        "ExactDuplicateGroup", back_populates="members"
    )
    image: Mapped["Image"] = relationship("Image")

    __table_args__ = (
        UniqueConstraint("group_id", "image_id", name="uq_member_group_image"),
    )


class NearDuplicateEdge(Base):
    """A pairwise near-duplicate relationship between two images."""

    __tablename__ = "near_duplicate_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_a_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    image_b_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    hamming_distance: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    method: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "phash", "orb", "combined"
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC), nullable=False
    )

    image_a: Mapped["Image"] = relationship(
        "Image", foreign_keys=[image_a_id], back_populates="near_dup_edges"
    )
    image_b: Mapped["Image"] = relationship("Image", foreign_keys=[image_b_id])

    __table_args__ = (
        UniqueConstraint("image_a_id", "image_b_id", name="uq_near_dup_pair"),
        Index("ix_near_dup_similarity", "similarity_score"),
        Index("ix_near_dup_method", "method"),
    )


class AIDetectionResult(Base):
    """AI-generated vs authentic classification result for one image."""

    __tablename__ = "ai_detection_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True, unique=True
    )
    label: Mapped[str] = mapped_column(String(32), nullable=False)  # authentic / ai_generated / uncertain
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    features_used: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC), nullable=False
    )

    image: Mapped["Image"] = relationship("Image", back_populates="ai_result")


class ForensicResult(Base):
    """Frequency-domain, noise, and compression forensic features for one image."""

    __tablename__ = "forensic_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True, unique=True
    )
    fft_energy_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dct_energy_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    noise_residual_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    compression_estimate: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    feature_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC), nullable=False
    )

    image: Mapped["Image"] = relationship("Image", back_populates="forensic")


class ScanHistory(Base):
    """Audit trail of scan operations."""

    __tablename__ = "scan_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    directory: Mapped[str] = mapped_column(String(1024), nullable=False)
    images_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    images_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    exact_groups: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    near_dup_pairs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    potential_savings_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC), nullable=False
    )
    completed_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class UserAction(Base):
    """Audit trail of user-initiated destructive / management actions."""

    __tablename__ = "user_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)  # delete, quarantine, restore
    image_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL"), nullable=True, index=True
    )
    source_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    destination_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC), nullable=False
    )
