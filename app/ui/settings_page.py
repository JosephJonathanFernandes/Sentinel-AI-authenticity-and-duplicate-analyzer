"""Settings page — configuration, quarantine management, model info."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from sentinel.app.config import settings, DATABASE_PATH, QUARANTINE_DIR, MODELS_DIR
from sentinel.app.core.file_manager import FileManager
from .common import format_size, init_session_state


def render_settings_page() -> None:
    st.title("Settings")
    st.markdown("Configure Sentinel paths, thresholds, and manage quarantined files.")

    st.subheader("Application paths")
    st.caption(f"Database: `{DATABASE_PATH}`")
    st.caption(f"Models directory: `{MODELS_DIR}`")
    st.caption(f"Quarantine directory: `{QUARANTINE_DIR}`")
    st.caption(f"Sample images: `{settings.SAMPLE_IMAGES_DIR}`")
    st.caption(f"Reports: `{settings.REPORTS_DIR}`")

    st.divider()

    st.subheader("Detection thresholds")
    st.number_input(
        "pHash / dHash / aHash size",
        min_value=4,
        max_value=32,
        value=settings.hash_size,
        disabled=True,
        help="Change in app/config.py and restart.",
    )
    st.number_input(
        "Near-duplicate combined similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.similarity_combined_threshold,
        step=0.05,
        disabled=True,
        help="Change in app/config.py and restart.",
    )
    st.number_input(
        "AI uncertainty threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        disabled=True,
        help="Change in app/ai_detection/inference.py and restart.",
    )
    st.caption("Thresholds are read from app/config.py at startup. Edit the file and restart the app to change them.")

    st.divider()

    st.subheader("Quarantine management")
    st.caption(
        "Files quarantined instead of deleted are moved here. "
        "You can restore them or permanently delete them from this page."
    )

    fm = FileManager()
    quarantined = fm.list_quarantined()
    q_size = fm.quarantine_size_bytes()

    c1, c2 = st.columns(2)
    c1.metric("Quarantined files", len(quarantined))
    c2.metric("Quarantine size", format_size(q_size))

    if quarantined:
        st.markdown("### Quarantined files")
        for qp in quarantined:
            col1, col2, col3 = st.columns([4, 2, 2])
            with col1:
                st.caption(qp.name)
                st.caption(str(qp))
            with col2:
                st.caption(format_size(qp.stat().st_size))
            with col3:
                if st.button("Restore", key=f"restore_{qp.name}", use_container_width=True):
                    result = fm.restore(qp, destination_dir=Path.cwd())
                    if result.success:
                        st.success(f"Restored to {result.destination_path}")
                    else:
                        st.error(f"Restore failed: {result.message}")
                    st.rerun()
                if st.button(
                    "Delete permanently",
                    key=f"delq_{qp.name}",
                    type="secondary",
                    use_container_width=True,
                ):
                    try:
                        qp.unlink()
                        st.success("Deleted from quarantine.")
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")
                    st.rerun()
            st.divider()
    else:
        st.info("No files in quarantine.")

    st.divider()

    st.subheader("Safety reminders")
    st.markdown(
        "- Files are **quarantined by default**, never permanently deleted without explicit action.\n"
        "- All destructive actions are logged to the database (`user_actions` table).\n"
        "- No images are uploaded to external servers; all processing is local.\n"
        "- AI detection results are never 100% certain — always review manually."
    )
