"""Duplicate pages — Exact Duplicates and Near Duplicates."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from sentinel.app.database.database import get_session
from sentinel.app.database import repository as repo
from sentinel.app.core.file_manager import FileManager
from sentinel.app.management.recommendations import RecommendationEngine
from .common import (
    format_size,
    format_dimensions,
    load_preview_image,
    status_badge,
    init_session_state,
)


def render_duplicate_page() -> None:
    """Render either Exact Duplicates or Near Duplicates based on query param."""
    # Determine which sub-section from sidebar label context is unreliable;
    # instead show both sections on one page with tabs.
    init_session_state()

    st.title("Duplicate Detection")
    tab_exact, tab_near = st.tabs(["Exact Duplicates", "Near Duplicates"])

    with tab_exact:
        _render_exact_duplicates()

    with tab_near:
        _render_near_duplicates()


def _render_exact_duplicates() -> None:
    st.subheader("Exact Duplicates (byte-identical, SHA-256)")
    st.caption(
        "Groups of images that are byte-for-byte identical. "
        "One copy is recommended as primary (highest resolution)."
    )

    engine = __import__("sentinel.app.ui.common", fromlist=["_get_engine"])._get_engine()
    with get_session(engine) as session:
        groups = repo.get_exact_groups(session)

    if not groups:
        st.info("No exact duplicate groups found. Run a scan first.")
        return

    total_savings = sum(g.redundant_size_bytes for g in groups)
    st.metric("Total redundant storage", format_size(total_savings))

    fm = FileManager()

    # Build recommendations for all exact groups using the engine.
    engine_rec = RecommendationEngine(session)
    group_recommendations = {
        g.id: engine_rec.recommend_for_exact_group(g) for g in groups
    }

    for g in groups:
        gr = group_recommendations[g.id]
        with st.expander(
            f"Group {g.id} — {g.file_count} images · "
            f"{format_size(g.total_size_bytes)} total · "
            f"{format_size(g.redundant_size_bytes)} redundant",
            expanded=False,
        ):
            members = sorted(g.members, key=lambda m: (not m.is_primary, m.image.file_name))

            # Show members in columns
            cols = st.columns(min(len(members), 4))
            for idx, m in enumerate(members):
                col = cols[idx % len(cols)]
                rec = next((r for r in gr.images if r.image.id == m.image.id), None)
                with col:
                    img = load_preview_image(m.image.path, max_size=200)
                    if img:
                        st.image(img, caption=m.image.file_name, use_container_width=True)
                    else:
                        st.caption(m.image.file_name)
                        st.caption("(preview unavailable)")

                    st.caption(format_dimensions(m.image.width, m.image.height))
                    st.caption(format_size(m.image.size_bytes))
                    if rec and rec.is_recommended:
                        status_badge("RECOMMENDED", "ok")
                    elif m.is_primary:
                        status_badge("PRIMARY", "info")
                    else:
                        status_badge("duplicate", "info")

                    if rec:
                        st.caption(f"Score: {rec.score:.1f}/100")
                        with st.expander("Why this score?", expanded=False):
                            for reason in rec.reasons:
                                st.caption(f"• {reason}")

                    # Action buttons
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if st.button(
                            "Quarantine", key=f"q_exact_{m.id}", use_container_width=True
                        ):
                            _quarantine_image(fm, m.image)
                    with b_col2:
                        if st.button(
                            "Delete permanently",
                            key=f"d_exact_{m.id}",
                            type="secondary",
                            use_container_width=True,
                        ):
                            _confirm_delete(fm, m.image)

            st.divider()

            # Group-level recommendation
            rec = gr.primary
            if rec:
                st.markdown(
                    f"**Recommended to keep:** `{rec.image.file_name}` "
                    f"(score {rec.score:.1f}/100) — "
                    f"{format_dimensions(rec.image.width, rec.image.height)}, "
                    f"{format_size(rec.image.size_bytes)}."
                )
                if rec.reasons:
                    st.caption("Reasons: " + "; ".join(rec.reasons))


def _render_near_duplicates() -> None:
    st.subheader("Near Duplicates (visually similar)")
    st.caption(
        "Pairs of images that look similar despite different file contents "
        "(resized, compressed, cropped, rotated, brightness/contrast adjusted, etc.). "
        "Matched using perceptual hashing + ORB feature matching."
    )

    engine = __import__("sentinel.app.ui.common", fromlist=["_get_engine"])._get_engine()
    with get_session(engine) as session:
        edges = repo.get_near_duplicate_edges(session, limit=500)

    if not edges:
        st.info("No near-duplicate pairs found. Run a scan first.")
        return

    st.metric("Near-duplicate pairs", len(edges))

    fm = FileManager()
    rec_engine = RecommendationEngine(session)

    for e in edges:
        gr = rec_engine.recommend_for_near_duplicate_pair(e.image_a, e.image_b)
        recommended = gr.primary

        with st.expander(
            f"Pair {e.id} — {e.method} · similarity {e.similarity_score:.0%}"
            + (f" (Hamming {e.hamming_distance})" if e.hamming_distance is not None else ""),
            expanded=False,
        ):
            col_a, col_b = st.columns(2)
            images = [(e.image_a, "Image A"), (e.image_b, "Image B")]
            for col, (img_obj, label) in zip(col_a, col_b):
                rec = next((r for r in gr.images if r.image.id == img_obj.id), None)
                with col:
                    is_rec = rec and rec.is_recommended
                    st.markdown(f"**{label}**")
                    prev = load_preview_image(img_obj.path, max_size=250)
                    if prev:
                        st.image(prev, caption=img_obj.file_name, use_container_width=True)
                    else:
                        st.caption(img_obj.file_name)
                        st.caption("(preview unavailable)")
                    st.caption(format_dimensions(img_obj.width, img_obj.height))
                    st.caption(format_size(img_obj.size_bytes))
                    st.caption(img_obj.path)

                    if is_rec:
                        status_badge("RECOMMENDED", "ok")
                    else:
                        status_badge("alternative", "info")

                    if rec:
                        st.caption(f"Score: {rec.score:.1f}/100")

                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button(
                            "Quarantine", key=f"q_near_{e.id}_{img_obj.id}",
                            use_container_width=True,
                        ):
                            _quarantine_image(fm, img_obj)
                    with b2:
                        if st.button(
                            "Delete permanently",
                            key=f"d_near_{e.id}_{img_obj.id}",
                            type="secondary",
                            use_container_width=True,
                        ):
                            _confirm_delete(fm, img_obj)

            st.divider()
            st.caption(
                f"Matching method: {e.method} · "
                f"Combined similarity score: {e.similarity_score:.2f}"
            )
            if recommended:
                st.caption(
                    f"🏆 Recommended to keep: **{recommended.image.file_name}** "
                    f"(score {recommended.score:.1f}/100)"
                )

    # Render any pending delete confirmation
    render_delete_confirmation()


def _quarantine_image(fm: FileManager, image) -> None:
    """Quarantine an image and record the action."""
    result = fm.quarantine(image.path)
    if result.success:
        st.success(f"Quarantined: {result.destination_path}")
    else:
        st.error(f"Quarantine failed: {result.message}")
    st.rerun()


def _confirm_delete(fm: FileManager, image) -> None:
    """Show a confirmation before permanent deletion."""
    st.session_state["_pending_delete"] = {
        "image_id": image.id,
        "path": image.path,
        "file_name": image.file_name,
    }
    st.rerun()


# Render pending delete confirmation if set
def render_delete_confirmation() -> None:
    """Render a delete confirmation dialog if one is pending."""
    pending = st.session_state.get("_pending_delete")
    if not pending:
        return

    st.warning(
        f"⚠️ Permanently delete **{pending['file_name']}**?\n\n"
        f"Path: `{pending['path']}`\n\n"
        "This action cannot be undone. Consider quarantining instead."
    )
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Yes, delete", type="secondary", key="confirm_delete_yes"):
            _execute_delete(pending)
    with col2:
        if st.button("Cancel", key="confirm_delete_no"):
            st.session_state["_pending_delete"] = None
            st.rerun()


def _execute_delete(pending: dict) -> None:
    from sentinel.app.database.database import get_session
    from sentinel.app.database import repository as repo
    from sentinel.app.core.file_manager import FileManager

    fm = FileManager()
    result = fm.delete_permanent(pending["path"])
    if result.success:
        st.success(f"Deleted: {pending['file_name']}")
    else:
        st.error(f"Delete failed: {result.message}")

    # Log the action in the DB
    engine = __import__("sentinel.app.ui.common", fromlist=["_get_engine"])._get_engine()
    with get_session(engine) as session:
        image = repo.get_image_by_path(session, pending["path"])
        repo.add_user_action(
            session,
            action_type="delete",
            image=image,
            source_path=pending["path"],
            details="Permanent deletion via dashboard",
        )
        session.commit()

    st.session_state["_pending_delete"] = None
    st.rerun()
