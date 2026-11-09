"""Sentinel dashboard — Streamlit entry point.

Provides sidebar navigation and routes to the individual page modules.
All processing is local; Streamlit calls the backend modules directly.
"""

from __future__ import annotations

import streamlit as st

from .common import init_session_state
from .scan_page import render_scan_page
from .duplicate_page import render_duplicate_page
from .ai_detection_page import render_ai_detection_page
from .forensic_page import render_forensic_page
from .reports_page import render_reports_page
from .settings_page import render_settings_page


def render_overview_page() -> None:
    """Render the Overview dashboard with summary cards."""
    st.title("SENTINEL — Overview")
    st.markdown(
        "Image authenticity and duplicate analysis dashboard. "
        "All processing runs locally on your machine."
    )

    from sentinel.app.database.database import get_session
    from sentinel.app.database import repository as repo

    engine = __import__("sentinel.app.ui.common", fromlist=["_get_engine"])._get_engine()
    with get_session(engine) as session:
        total_images = repo.count_images(session)
        by_ext = repo.count_by_extension(session)
        exact_groups = repo.get_exact_groups(session)
        near_edges = repo.get_near_duplicate_edges(session, limit=10000)
        ai_results = repo.get_ai_results(session, limit=10000)

        savings = sum(g.redundant_size_bytes for g in exact_groups)
        savings_mb = savings / (1024 * 1024)

        ai_generated = sum(1 for r in ai_results if r.label == "ai_generated")
        ai_authentic = sum(1 for r in ai_results if r.label == "authentic")
        ai_uncertain = sum(1 for r in ai_results if r.label == "uncertain")

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total images scanned", total_images)
    c2.metric("Exact duplicate groups", len(exact_groups))
    c3.metric("Near-duplicate pairs", len(near_edges))
    c4.metric("Potential storage savings", f"{savings_mb:.2f} MB")

    c5, c6, c7 = st.columns(3)
    c5.metric("Likely AI-generated", ai_generated)
    c6.metric("Likely authentic", ai_authentic)
    c7.metric("Inconclusive", ai_uncertain)

    st.divider()

    # Quick actions
    st.subheader("Quick actions")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Scan a folder for images", use_container_width=True):
            st.session_state["_nav_target"] = "Scan Images"
            st.rerun()
    with col_b:
        if st.button("View exact duplicates", use_container_width=True):
            st.session_state["_nav_target"] = "Exact Duplicates"
            st.rerun()

    if by_ext:
        st.divider()
        st.subheader("Images by file type")
        st.bar_chart(by_ext)

    # Recent AI results
    if ai_results:
        st.divider()
        st.subheader("Recent AI detection results")
        rows = []
        for r in ai_results[:10]:
            rows.append(
                {
                    "Image": r.image.file_name,
                    "Label": r.label.replace("_", " ").title(),
                    "Confidence": f"{r.confidence:.0%}",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)


PAGES = {
    "Overview": render_overview_page,
    "Scan Images": render_scan_page,
    "Exact Duplicates": render_duplicate_page,
    "Near Duplicates": render_duplicate_page,
    "AI Image Detection": render_ai_detection_page,
    "Image Forensics": render_forensic_page,
    "Reports": render_reports_page,
    "Settings": render_settings_page,
}


def main() -> None:
    st.set_page_config(
        page_title="SENTINEL — Image Authenticity & Duplicate Analyzer",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom dark-theme-friendly styling
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        .stMetric { background: rgba(128,128,128,0.06); padding: 12px; border-radius: 8px; }
        h1 { font-weight: 700; letter-spacing: -0.02em; }
        .section-title { font-size: 1.25rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    # Handle programmatic navigation
    default_index = 0
    if "_nav_target" in st.session_state:
        target = st.session_state.pop("_nav_target")
        if target in PAGES:
            default_index = list(PAGES.keys()).index(target)

    st.sidebar.title("🛡️ SENTINEL")
    st.sidebar.markdown("Image Authenticity & Duplicate Analyzer")
    st.sidebar.divider()

    selection = st.sidebar.radio(
        "Navigation",
        list(PAGES.keys()),
        index=default_index,
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption("Local-first · CPU-only · No cloud uploads")
    st.sidebar.caption(f"Database: {DATABASE_PATH.name}")

    # Route to the selected page
    page_fn = PAGES[selection]
    page_fn()


if __name__ == "__main__":
    main()
