"""Sentinel demonstration script.

Creates a synthetic image folder with duplicates and near-duplicates, runs
the full detection pipeline, and prints a summary.  Use this to quickly
verify the application works end-to-end without any external data.

Usage:

    python demo.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PIL import Image

from sentinel.app.database.database import create_engine_for, init_db, get_session
from sentinel.app.core.scanner import Scanner
from sentinel.app.core.image_loader import get_basic_info
from sentinel.app.database import repository as repo
from sentinel.app.duplicate_detection.exact_hash import ExactDuplicateEngine
from sentinel.app.duplicate_detection.similarity_engine import SimilarityEngine
from sentinel.app.management.recommendations import RecommendationEngine
import datetime


def _make_variants(folder: Path) -> None:
    """Create a small synthetic image collection with duplicates/near-dups."""
    folder.mkdir(parents=True, exist_ok=True)

    # Base image
    base = Image.new("RGB", (400, 400), (70, 130, 180))

    # Exact duplicates (same bytes)
    base.save(folder / "vacation.jpg", "JPEG", quality=90)
    shutil.copy(folder / "vacation.jpg", folder / "vacation_copy.jpg")

    # Near-duplicate: resized
    base.resize((200, 200)).save(folder / "vacation_small.jpg", "JPEG", quality=85)

    # Near-duplicate: re-encoded at different quality
    base.save(folder / "vacation_lowq.jpg", "JPEG", quality=60)

    # Unrelated image
    Image.new("RGB", (400, 400), (200, 50, 50)).save(folder / "sunset.jpg", "JPEG", quality=90)

    # Corrupted file
    (folder / "corrupt.jpg").write_bytes(b"not an image")


def main() -> None:
    print("=" * 60)
    print("SENTINEL End-to-End Demonstration")
    print("=" * 60)
    print()

    tmp = Path(tempfile.mkdtemp(prefix="sentinel_demo_"))
    img_dir = tmp / "photos"
    _make_variants(img_dir)

    db_path = tmp / "sentinel.db"
    engine = create_engine_for(db_path)
    init_db(engine)

    print(f"Scanning: {img_dir}")
    print(f"Database: {db_path}")
    print()

    with get_session(engine) as session:
        scanner = Scanner()
        for p in scanner.discover(img_dir):
            info = get_basic_info(p)
            if info["is_corrupted"]:
                print(f"  [SKIP] {p.name} — {info['error_message']}")
                continue
            repo.create_image(
                session,
                {
                    "path": str(p),
                    "file_name": p.name,
                    "directory": str(p.parent),
                    "extension": p.suffix.lower(),
                    "size_bytes": info["size_bytes"],
                    "width": info["width"],
                    "height": info["height"],
                    "color_mode": info["mode"],
                    "format": info["format"],
                    "scanned_at": datetime.datetime.now(datetime.UTC),
                },
            )
        session.commit()
        indexed = repo.count_images(session)
        print(f"Indexed {indexed} image(s).")
        print()

        print("--- Exact Duplicate Groups ---")
        groups = ExactDuplicateEngine(session).build_groups(min_group_size=2)
        for g in groups:
            print(f"  Group {g.id}: {g.file_count} files, "
                  f"{g.redundant_size_bytes} bytes redundant")
            for m in g.members:
                print(f"    - {m.image.file_name} "
                      f"({m.image.width}x{m.image.height}, {m.image.size_bytes} B)"
                      f"{' [PRIMARY]' if m.is_primary else ''}")
        if not groups:
            print("  (none)")
        print()

        print("--- Near-Duplicate Pairs ---")
        edges = SimilarityEngine(session).find_near_duplicates()
        for e in edges:
            print(f"  {e.image_a.file_name} <-> {e.image_b.file_name}")
            print(f"      method={e.method} similarity={e.similarity_score:.2f}")
        if not edges:
            print("  (none)")
        print()

        print("--- Retention Recommendations ---")
        rec_engine = RecommendationEngine(session)
        for g in groups:
            gr = rec_engine.recommend_for_exact_group(g)
            r = gr.primary
            print(f"  Group {g.id}: keep '{r.image.file_name}' "
                  f"(score {r.score:.1f}/100)")
            print(f"      Reasons: {'; '.join(r.reasons)}")
        for e in edges:
            gr = rec_engine.recommend_for_near_duplicate_pair(e.image_a, e.image_b)
            r = gr.primary
            print(f"  Pair {e.id}: keep '{r.image.file_name}' "
                  f"(score {r.score:.1f}/100)")
        print()

    # Cleanup
    engine.dispose()
    shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 60)
    print("Demonstration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
