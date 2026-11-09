"""Image Forensics page — metadata, FFT/DCT, noise analysis."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from sentinel.app.forensics.forensic_engine import ForensicEngine
from sentinel.app.core.image_loader import get_basic_info
from .common import (
    format_size,
    format_dimensions,
    load_preview_image,
    status_badge,
    init_session_state,
)


def render_forensic_page() -> None:
    st.title("Image Forensics")
    st.markdown(
        "Analyze a single image for forensic signals: metadata, frequency-domain "
        "(FFT/DCT) characteristics, and noise-residual patterns. "
        "All signals are **supporting evidence only** — none proves AI generation."
    )

    img_path = st.text_input(
        "Image path",
        placeholder="e.g. C:\\Users\\Joseph\\Pictures\\photo.jpg",
        help="Full path to a single image file.",
    )
    run_btn = st.button("Analyze", type="primary", use_container_width=True)

    if run_btn:
        if not img_path:
            st.error("Please enter an image path.")
            return
        p = Path(img_path)
        if not p.exists():
            st.error(f"File not found: {img_path}")
            return
        _run_forensics(p)

    st.divider()
    st.caption(
        "Missing or unusual metadata does NOT prove an image is AI-generated. "
        "Frequency and noise anomalies are heuristics, not definitive classification."
    )


def _run_forensics(path: Path) -> None:
    """Run the full forensic pipeline on one image and display results."""
    engine = ForensicEngine()

    with st.spinner("Running forensic analysis..."):
        try:
            report = engine.analyze(path)
        except Exception as exc:
            st.error(f"Forensic analysis failed: {exc}")
            return

    # Header info
    st.subheader(f"Forensic report: {report.file_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Size", format_size(report.size_bytes))
    c2.metric("Extension", report.extension.upper())
    c3.metric("Composite anomaly score", f"{report.composite_score:.2f}")

    if report.errors:
        with st.expander("Errors / warnings", expanded=False):
            for e in report.errors:
                st.caption(f"⚠️ {e}")

    # Metadata section
    st.divider()
    st.subheader("Metadata")
    meta = report.metadata
    if meta and not meta.is_corrupted:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Dimensions", format_dimensions(meta.width, meta.height))
        mc2.metric("Format", meta.format or "—")
        mc3.metric("Color mode", meta.color_mode or "—")
        mc4.metric("MIME", meta.mime_type or "—")

        mrows = [
            ("Camera make", meta.camera_make or "—"),
            ("Camera model", meta.camera_model or "—"),
            ("Software", meta.software or "—"),
            ("Date time", meta.date_time or "—"),
            ("Date original", meta.date_time_original or "—"),
            ("Metadata complete", "Yes" if meta.metadata_complete else "No"),
            ("Suspicious", "Yes" if meta.suspicious else "No"),
        ]
        for label, value in mrows:
            colx, coly = st.columns([2, 4])
            colx.markdown(f"**{label}**")
            if label == "Suspicious" and value == "Yes":
                coly.warning(value)
            else:
                coly.caption(value)

        if meta.suspicious_reason:
            st.caption(f"Reason: {meta.suspicious_reason}")
    else:
        st.info("No usable metadata extracted.")

    # Frequency section
    st.divider()
    st.subheader("Frequency analysis (FFT / DCT)")
    freq = report.frequency
    if freq and not freq.error:
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("FFT low-freq energy", f"{freq.fft_energy_ratio_low or 0:.2%}")
        fc2.metric("FFT mid-freq energy", f"{freq.fft_energy_ratio_mid or 0:.2%}")
        fc3.metric("FFT high-freq energy", f"{freq.fft_energy_ratio_high or 0:.2%}")

        fd1, fd2 = st.columns(2)
        fd1.metric("Spectral flatness", f"{freq.fft_spectral_flatness or 0:.3f}")
        fd2.metric("Compression indicator", freq.compression_indicator or "—")

        frows = [
            ("DCT low-freq energy", f"{freq.dct_energy_ratio_low or 0:.2%}"),
            ("DCT mid-freq energy", f"{freq.dct_energy_ratio_mid or 0:.2%}"),
            ("DCT high-freq energy", f"{freq.dct_energy_ratio_high or 0:.2%}"),
        ]
        for label, value in frows:
            colx, coly = st.columns([2, 4])
            colx.markdown(f"**{label}**")
            coly.caption(value)

        st.caption(
            f"Frequency anomaly score: {report.frequency_anomaly_score:.2f} "
            "(higher = more anomalous frequency pattern)"
        )
    else:
        st.info("Frequency analysis unavailable.")
        if freq and freq.error:
            st.caption(freq.error)

    # Noise section
    st.divider()
    st.subheader("Noise-residual analysis")
    noise = report.noise
    if noise and not noise.error:
        nc1, nc2, nc3 = st.columns(3)
        nc1.metric("Residual std", f"{noise.noise_residual_std or 0:.4f}")
        nc2.metric("Residual mean", f"{noise.noise_residual_mean or 0:.4f}")
        nc3.metric("Residual skew", f"{noise.noise_residual_skew or 0:.3f}")

        nd1, nd2 = st.columns(2)
        nd1.metric("Residual kurtosis", f"{noise.noise_residual_kurtosis or 0:.3f}")
        nd2.metric("Noise consistency", f"{noise.noise_consistency_score or 0:.2f}")

        nrows = [
            ("Local variance mean", f"{noise.noise_local_variance_mean or 0:.6f}"),
            ("Local variance std", f"{noise.noise_local_variance_std or 0:.6f}"),
        ]
        for label, value in nrows:
            colx, coly = st.columns([2, 4])
            colx.markdown(f"**{label}**")
            coly.caption(value)

        st.caption(
            f"Noise anomaly score: {report.noise_anomaly_score:.2f} "
            "(higher = more anomalous noise pattern)"
        )
    else:
        st.info("Noise analysis unavailable.")
        if noise and noise.error:
            st.caption(noise.error)

    # Interpretation
    st.divider()
    st.subheader("Interpretation")
    if report.composite_score >= 0.5:
        st.warning(
            f"Composite forensic anomaly score is {report.composite_score:.2f} — "
            "several signals are unusual. Treat as supporting evidence only; "
            "not proof of AI generation."
        )
    else:
        st.info(
            f"Composite forensic anomaly score is {report.composite_score:.2f} — "
            "no strong forensic anomalies detected."
        )
