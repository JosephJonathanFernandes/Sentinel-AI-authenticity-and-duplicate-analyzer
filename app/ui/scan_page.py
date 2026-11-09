"""Scan Images page — directory selection, scanning, and progress."""

from __future__ import annotations

from pathlib import Path
import datetime

import streamlit as st

from sentinel.app.config import settings
from sentinel.app.core.scanner import Scanner
from sentinel.app.core.image_loader import get_basic_info
from sentinel.app.database.database import get_session, init_db
from sentinel.app.database import repository as repo
from sentinel.app.duplicate_detection.exact_hash import ExactDuplicateEngine
from sentinel.app.duplicate_detection.similarity_engine import SimilarityEngine
from .common import format_size, format_dimensions, load_preview_image


def render_scan_page() -> None:
    st.title("Scan Images")
    st.markdown(
        "Select a folder to scan. Sentinel will index all images, compute "
        "SHA-256 hashes, perceptual hashes, and ORB features, then detect "
        "exact and near-duplicates. Processing happens locally."
    )

    # Directory input
    col1, col2 = st.columns([4, 1])
    with col1:
        directory = st.text_input(
            "Folder path",
            value=st.session_state.get("scan_directory", ""),
            placeholder="e.g. C:\\Users\\Joseph\\Pictures",
            help="Enter or paste the full path to an image folder.",
        )
    with col2:
        st.write("")
        scan_btn = st.button("Start scan", type="primary", use_container_width=True)

    if directory:
        st.session_state["scan_directory"] = directory
        path = Path(directory)
        if not path.exists():
            st.error(f"Directory not found: {directory}")
            return
        if not path.is_dir():
            st.error(f"Not a directory: {directory}")
            return

        # Show existing images count from DB
        engine = __import__("sentinel.app.ui.common", fromlist=["_get_engine"])._get_engine()
        with get_session(engine) as session:
            existing = repo.count_images(session)
        st.info(f"Database already contains {existing} indexed image(s).")

    if scan_btn:
        if not directory:
            st.error("Please enter a folder path first.")
            return
        _run_scan(directory)


def _run_scan(directory: str) -> None:
    """Execute the scan and duplicate detection pipeline."""
    path = Path(directory)
    engine = __import__("sentinel.app.ui.common", fromlist=["_get_engine"])._get_engine()
    init_db(engine)

    import uuid as _uuid
    scan_id = str(_uuid.uuid4())

    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_cols = st.columns(4)
    found_metric = metrics_cols[0].empty()
    processed_metric = metrics_cols[1].empty()
    exact_metric = metrics_cols[2].empty()
    near_metric = metrics_cols[3].empty()

    st.session_state["scan_running"] = True
    st.session_state["scan_progress"] = 0
    st.session_state["scan_total"] = 0
    st.session_state["scan_status"] = "running"
    st.session_state["last_scan_id"] = scan_id

    try:
        with get_session(engine) as session:
            history = repo.create_scan_history(
                session,
                {
                    "scan_id": scan_id,
                    "directory": str(path),
                    "status": "running",
                },
            )

            scanner = Scanner()
            found = 0
            processed = 0

            # Phase 1: discover and index images
            status_text.text("Discovering images...")
            all_paths = list(scanner.discover(path))
            total = len(all_paths)
            st.session_state["scan_total"] = total
            found_metric.metric("Images found", total)

            for idx, p in enumerate(all_paths):
                found += 1
                info = get_basic_info(p)
                if info["is_corrupted"]:
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
                processed += 1

                # Update progress every 10 files
                if found % 10 == 0 or found == total:
                    pct = min(int((found / total) * 100), 100) if total else 100
                    progress_bar.progress(pct)
                    status_text.text(
                        f"Indexing {found}/{total} images ({processed} valid)"
                    )
                    processed_metric.metric("Images processed", processed)

            session.commit()

            # Phase 2: exact duplicates
            status_text.text("Detecting exact duplicates...")
            exact_groups = ExactDuplicateEngine(session).build_groups(min_group_size=2)
            exact_metric.metric("Exact groups", len(exact_groups))

            # Phase 3: near duplicates
            status_text.text("Detecting near-duplicates...")
            near_edges = SimilarityEngine(session).find_near_duplicates()
            near_metric.metric("Near-dup pairs", len(near_edges))

            savings = sum(g.redundant_size_bytes for g in exact_groups)

            repo.update_scan_history(
                session,
                history,
                {
                    "images_found": found,
                    "images_processed": processed,
                    "exact_groups": len(exact_groups),
                    "near_dup_pairs": len(near_edges),
                    "potential_savings_bytes": savings,
                    "completed_at": datetime.datetime.now(datetime.UTC),
                    "status": "completed",
                },
            )
            session.commit()

        progress_bar.progress(100)
        status_text.text("Scan complete.")
        st.session_state["scan_running"] = False
        st.session_state["scan_status"] = "completed"

        st.success(
            f"Scan finished: {processed} images indexed, "
            f"{len(exact_groups)} exact duplicate groups, "
            f"{len(near_edges)} near-duplicate pairs, "
            f"{format_size(savings)} potential savings."
        )

        # Navigation hint
        if st.button("View exact duplicates", type="secondary"):
            st.session_state["_nav_target"] = "Exact Duplicates"
            st.rerun()

    except Exception as exc:  # pragma: no cover - defensive UI safety net
        st.session_state["scan_running"] = False
        st.session_state["scan_status"] = "failed"
        st.error(f"Scan failed: {exc}")
        try:
            with get_session(engine) as session:
                repo.update_scan_history(
                    session,
                    history,
                    {"status": "failed", "error_message": str(exc)},
                )
                session.commit()
        except Exception:
            pass
