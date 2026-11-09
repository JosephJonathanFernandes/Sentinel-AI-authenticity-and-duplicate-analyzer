"""Sentinel FastAPI application.

Provides a small REST API backing the Streamlit dashboard. All processing is
local; no image bytes are sent over the network beyond the local loopback.
Endpoints are intentionally limited to scan / duplicate / forensic / AI /
management operations on user-selected directories.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import settings, DATABASE_PATH
from .database.database import create_engine_for, init_db, get_session
from .database import repository as repo
from .core.scanner import Scanner
from .core.image_loader import get_basic_info
from .duplicate_detection.exact_hash import ExactDuplicateEngine
from .duplicate_detection.similarity_engine import SimilarityEngine

# ---------------------------------------------------------------------------
# Application state (engine instances are created at startup)
# ---------------------------------------------------------------------------
_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = create_engine_for(DATABASE_PATH)
    init_db(_engine)
    yield
    _engine = None


app = FastAPI(
    title="Sentinel — AI Authenticity & Duplicate Image Analyzer",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_engine():
    if _engine is None:
        raise RuntimeError("Database engine not initialised")
    return _engine


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ScanRequest(BaseModel):
    directory: str


class ScanResponse(BaseModel):
    scan_id: str
    status: str
    images_found: int
    images_processed: int
    exact_groups: int
    near_duplicate_pairs: int
    potential_savings_bytes: int


class DuplicateGroupOut(BaseModel):
    group_id: int
    sha256: str
    file_count: int
    total_size_bytes: int
    redundant_size_bytes: int
    members: list[dict]


class NearDuplicatePairOut(BaseModel):
    edge_id: int
    image_a: dict
    image_b: dict
    similarity_score: float
    hamming_distance: Optional[int]
    method: str


class AIResultOut(BaseModel):
    image_id: int
    label: str
    confidence: float
    model_version: Optional[str]
    explanation: Optional[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "sentinel"}


@app.post("/scan", response_model=ScanResponse)
def scan_images(request: ScanRequest):
    directory = Path(request.directory)
    if not directory.exists() or not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Invalid directory: {directory}")

    import uuid as _uuid
    scan_id = str(_uuid.uuid4())

    engine = _get_engine()
    with get_session(engine) as session:
        history = repo.create_scan_history(
            session,
            {
                "scan_id": scan_id,
                "directory": str(directory),
                "status": "running",
            },
        )

        scanner = Scanner()
        found = 0
        processed = 0
        exact_engine = ExactDuplicateEngine(session)
        sim_engine = SimilarityEngine(session)

        try:
            for batch in scanner.batch_paths(directory, settings.batch_size):
                for path in batch:
                    found += 1
                    info = get_basic_info(path)
                    if info["is_corrupted"]:
                        continue

                    image = repo.create_image(
                        session,
                        {
                            "path": info["path"],
                            "file_name": info["file_name"],
                            "directory": info["directory"],
                            "extension": info["extension"],
                            "size_bytes": info["size_bytes"],
                            "width": info["width"],
                            "height": info["height"],
                            "color_mode": info["mode"],
                            "format": info["format"],
                            "scanned_at": __import__("datetime").datetime.utcnow(),
                        },
                    )
                    processed += 1

                # Commit incrementally so a crash doesn't lose everything.
                session.commit()

            # Run exact duplicate detection over everything just scanned.
            exact_groups = exact_engine.build_groups()
            near_pairs = sim_engine.find_near_duplicates()

            savings = sum(g.redundant_size_bytes for g in exact_groups)

            repo.update_scan_history(
                session,
                history,
                {
                    "images_found": found,
                    "images_processed": processed,
                    "exact_groups": len(exact_groups),
                    "near_dup_pairs": len(near_pairs),
                    "potential_savings_bytes": savings,
                    "completed_at": __import__("datetime").datetime.utcnow(),
                    "status": "completed",
                },
            )
            session.commit()
        except Exception as exc:  # pragma: no cover - safety net
            repo.update_scan_history(
                session, history, {"status": "failed", "error_message": str(exc)}
            )
            session.commit()
            raise HTTPException(status_code=500, detail=str(exc))

    return ScanResponse(
        scan_id=scan_id,
        status="completed",
        images_found=found,
        images_processed=processed,
        exact_groups=len(exact_groups),
        near_duplicate_pairs=len(near_pairs),
        potential_savings_bytes=savings,
    )


@app.get("/duplicates/exact", response_model=list[DuplicateGroupOut])
def list_exact_duplicates():
    engine = _get_engine()
    with get_session(engine) as session:
        groups = repo.get_exact_groups(session)
        result = []
        for g in groups:
            members = [
                {
                    "member_id": m.id,
                    "image_id": m.image_id,
                    "path": m.image.path,
                    "file_name": m.image.file_name,
                    "size_bytes": m.image.size_bytes,
                    "width": m.image.width,
                    "height": m.image.height,
                    "is_primary": m.is_primary,
                }
                for m in g.members
            ]
            result.append(
                DuplicateGroupOut(
                    group_id=g.id,
                    sha256=g.sha256,
                    file_count=g.file_count,
                    total_size_bytes=g.total_size_bytes,
                    redundant_size_bytes=g.redundant_size_bytes,
                    members=members,
                )
            )
        return result


@app.get("/duplicates/near", response_model=list[NearDuplicatePairOut])
def list_near_duplicates(limit: int = Query(200, ge=1, le=1000)):
    engine = _get_engine()
    with get_session(engine) as session:
        edges = repo.get_near_duplicate_edges(session, limit=limit)
        result = []
        for e in edges:
            result.append(
                NearDuplicatePairOut(
                    edge_id=e.id,
                    image_a={
                        "image_id": e.image_a.id,
                        "path": e.image_a.path,
                        "file_name": e.image_a.file_name,
                    },
                    image_b={
                        "image_id": e.image_b.id,
                        "path": e.image_b.path,
                        "file_name": e.image_b.file_name,
                    },
                    similarity_score=e.similarity_score,
                    hamming_distance=e.hamming_distance,
                    method=e.method,
                )
            )
        return result


@app.get("/dashboard/summary")
def dashboard_summary():
    engine = _get_engine()
    with get_session(engine) as session:
        total_images = repo.count_images(session)
        by_ext = repo.count_by_extension(session)
        exact_groups = repo.get_exact_groups(session)
        near_edges = repo.get_near_duplicate_edges(session, limit=10000)
        savings = sum(g.redundant_size_bytes for g in exact_groups)
        ai_results = repo.get_ai_results(session, limit=10000)

        ai_generated = sum(1 for r in ai_results if r.label == "ai_generated")
        inconclusive = sum(1 for r in ai_results if r.label == "uncertain")

    return {
        "total_images": total_images,
        "by_extension": by_ext,
        "exact_duplicate_groups": len(exact_groups),
        "near_duplicate_pairs": len(near_edges),
        "potential_savings_bytes": savings,
        "potential_savings_mb": round(savings / (1024 * 1024), 2),
        "ai_generated_count": ai_generated,
        "inconclusive_count": inconclusive,
    }


@app.get("/images")
def list_images(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    engine = _get_engine()
    with get_session(engine) as session:
        images = repo.list_images(session, limit=limit, offset=offset)
        return [
            {
                "id": i.id,
                "path": i.path,
                "file_name": i.file_name,
                "extension": i.extension,
                "size_bytes": i.size_bytes,
                "width": i.width,
                "height": i.height,
                "format": i.format,
                "sha256": i.sha256,
                "is_corrupted": i.is_corrupted,
            }
            for i in images
        ]
