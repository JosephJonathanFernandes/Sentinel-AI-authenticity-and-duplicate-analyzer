"""Shared UI helpers for the Sentinel Streamlit dashboard.

Provides database access, image display utilities, and formatting helpers
used across all dashboard pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from sentinel.app.config import DATABASE_PATH, QUARANTINE_DIR, settings
from sentinel.app.database.database import create_engine_for, init_db, get_session
from sentinel.app.database import repository as repo


# ---------------------------------------------------------------------------
# Database helper (cached engine)
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_engine():
    """Create and cache a SQLAlchemy engine for the Streamlit session."""
    engine = create_engine_for(DATABASE_PATH)
    init_db(engine)
    return engine


def get_db_session():
    """Yield a database session for the current Streamlit run."""
    engine = _get_engine()
    return get_session(engine)


# ---------------------------------------------------------------------------
# Image display helpers
# ---------------------------------------------------------------------------
def load_preview_image(path: str | Path, max_size: int = 400) -> Optional[Image.Image]:
    """Load an image for preview, resizing to fit within max_size.

    Returns None if the image cannot be loaded.
    """
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size))
        return img
    except Exception:
        return None


def format_size(num_bytes: Optional[int]) -> str:
    """Format a byte count into a human-readable string."""
    if num_bytes is None:
        return "Unknown"
    num_bytes = int(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def format_dimensions(width: Optional[int], height: Optional[int]) -> str:
    """Format image dimensions as 'W x H'."""
    if width is None or height is None:
        return "Unknown"
    return f"{width} × {height}"


def status_badge(label: str, kind: str = "info") -> None:
    """Render a small coloured badge for a status value."""
    colors = {
        "ok": "green",
        "warn": "orange",
        "error": "red",
        "info": "blue",
        "ai": "purple",
        "authentic": "green",
        "uncertain": "orange",
    }
    color = colors.get(kind, "gray")
    st.markdown(
        f"<span style='background-color:{color};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.8em;'>{label}</span>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Navigation state
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialise Streamlit session state keys used across pages."""
    defaults = {
        "scan_directory": "",
        "scan_running": False,
        "scan_progress": 0,
        "scan_total": 0,
        "scan_status": "idle",
        "last_scan_id": None,
        "selected_duplicate_group": None,
        "quarantine_confirm": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
