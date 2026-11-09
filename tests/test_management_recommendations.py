"""Tests for the image retention recommendation engine.

Verifies scoring, ranking, and explanation generation for duplicate groups.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.app.database.database import create_engine_for, init_db, get_session
from sentinel.app.database import models, repository as repo
from sentinel.app.management.recommendations import RecommendationEngine
from sentinel.app.core.image_loader import get_basic_info


def _store_image(session, path: Path) -> models.Image:
    info = get_basic_info(path)
    return repo.create_image(
        session,
        {
            "path": str(path),
            "file_name": path.name,
            "directory": str(path.parent),
            "extension": path.suffix.lower(),
            "size_bytes": info["size_bytes"],
            "width": info["width"],
            "height": info["height"],
            "color_mode": info["mode"],
            "format": info["format"],
        },
    )


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine_for(db_path)
    init_db(engine)
    with get_session(engine) as session:
        yield session
    engine.dispose()


def test_rank_prefers_higher_resolution(tmp_path):
    """Higher-resolution copy should rank first."""
    from PIL import Image

    low = tmp_path / "low.png"
    high = tmp_path / "high.png"
    Image.new("RGB", (100, 100), (50, 100, 150)).save(low, "PNG")
    Image.new("RGB", (400, 400), (50, 100, 150)).save(high, "PNG")

    db_path = tmp_path / "rec_test.db"
    engine = create_engine_for(db_path)
    init_db(engine)
    with get_session(engine) as session:
        img_low = _store_image(session, low)
        img_high = _store_image(session, high)
        session.commit()

        eng = RecommendationEngine(session)
        grp = eng.recommend_for_near_duplicate_pair(img_low, img_high)

        assert grp.recommended_image.id == img_high.id
        assert grp.images[0].is_recommended is True
        assert grp.images[0].image.id == img_high.id
        assert grp.images[1].is_recommended is False
        assert "Highest resolution in the group" in grp.images[0].reasons or "Good resolution" in grp.images[0].reasons


def test_rank_scores_in_0_to_100(tmp_path):
    from PIL import Image

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (200, 200), (10, 20, 30)).save(a, "PNG")
    Image.new("RGB", (200, 200), (40, 50, 60)).save(b, "PNG")

    db_path = tmp_path / "rec_test2.db"
    engine = create_engine_for(db_path)
    init_db(engine)
    with get_session(engine) as session:
        img_a = _store_image(session, a)
        img_b = _store_image(session, b)
        session.commit()

        eng = RecommendationEngine(session)
        grp = eng.recommend_for_near_duplicate_pair(img_a, img_b)

        for rec in grp.images:
            assert 0.0 <= rec.score <= 100.0, f"Score out of range: {rec.score}"
            assert rec.rank in (1, 2)
            assert len(rec.reasons) > 0


def test_exact_group_recommendation(tmp_path):
    from PIL import Image

    # Create two byte-identical images with different sizes (impossible if
    # truly byte-identical, so just use two images in a mock group).
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (300, 300), (100, 100, 100)).save(a, "PNG")
    Image.new("RGB", (100, 100), (200, 200, 200)).save(b, "PNG")

    db_path = tmp_path / "rec_test3.db"
    engine = create_engine_for(db_path)
    init_db(engine)
    with get_session(engine) as session:
        img_a = _store_image(session, a)
        img_b = _store_image(session, b)
        session.commit()

        # Build a mock exact group
        group = repo.create_exact_group(
            session, sha256="fake", file_count=2,
            total_size_bytes=img_a.size_bytes + img_b.size_bytes,
            redundant_size_bytes=img_b.size_bytes,
        )
        repo.add_exact_member(session, group, img_a, is_primary=False)
        repo.add_exact_member(session, group, img_b, is_primary=False)
        session.commit()

        eng = RecommendationEngine(session)
        grp = eng.recommend_for_exact_group(group)

        assert grp.group_type == "exact"
        assert grp.group_id == group.id
        assert len(grp.images) == 2
        assert grp.recommended_image is not None
        assert grp.primary.is_recommended is True
