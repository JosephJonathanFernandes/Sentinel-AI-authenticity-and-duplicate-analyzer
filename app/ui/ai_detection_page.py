"""AI Image Detection page — classification results and inference."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from sentinel.app.config import settings
from sentinel.app.database.database import get_session
from sentinel.app.database import repository as repo
from sentinel.app.ai_detection.inference import AIClassifier, LABEL_UNCERTAIN
from sentinel.app.ai_detection.explainability import GradCAMExplainer
from .common import (
    format_size,
    format_dimensions,
    load_preview_image,
    status_badge,
    init_session_state,
)


def render_ai_detection_page() -> None:
    st.title("AI Image Detection")
    st.markdown(
        "Classify images as **likely authentic**, **likely AI-generated**, or "
        "**uncertain**. A model must be trained first (see Settings or run "
        "`python train_ai.py`). All inference runs locally on CPU."
    )

    # Model status
    model_path = Path(settings.ai_model_path)
    model_exists = model_path.exists()

    c1, c2, c3 = st.columns(3)
    c1.metric("Model file", "Present" if model_exists else "Not trained")
    c2.metric("Input size", f"{settings.ai_image_size} × {settings.ai_image_size}")
    c3.metric("Uncertainty threshold", f"{0.60:.0%}")

    if not model_exists:
        st.warning(
            "No trained model found at `models/ai_classifier.pt`. "
            "Train one with:\n\n`python train_ai.py`\n\n"
            "Note: the default training uses procedurally generated synthetic "
            "images. For research-grade results, point at a real dataset "
            "(e.g. CIFAKE) via `sentinel/app/ai_detection/dataset.py`."
        )
        st.divider()

    # Show existing AI results
    st.subheader("Classification results")
    engine = __import__("sentinel.app.ui.common", fromlist=["_get_engine"])._get_engine()
    with get_session(engine) as session:
        results = repo.get_ai_results(session, limit=200)

    if not results:
        st.info("No AI detection results yet. Run inference below or scan images first.")
    else:
        # Summary counts
        ai_gen = sum(1 for r in results if r.label == "ai_generated")
        auth = sum(1 for r in results if r.label == "authentic")
        unc = sum(1 for r in results if r.label == "uncertain")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Likely AI-generated", ai_gen)
        cc2.metric("Likely authentic", auth)
        cc3.metric("Inconclusive", unc)

        st.divider()

        # Results table
        rows = []
        for r in results:
            rows.append(
                {
                    "Image": r.image.file_name,
                    "Label": r.label.replace("_", " ").title(),
                    "Confidence": f"{r.confidence:.0%}",
                    "AI probability": f"{0.0 if r.label=='uncertain' else r.ai_probability if hasattr(r,'ai_probability') else r.confidence:.0%}",
                    "Path": r.image.path,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()

    # Run inference on a single image
    st.subheader("Run inference on an image")
    img_path = st.text_input(
        "Image path",
        placeholder="e.g. C:\\Users\\Joseph\\Pictures\\photo.jpg",
        help="Full path to a single image file.",
    )
    col_run, col_clear = st.columns([1, 4])
    with col_run:
        run_btn = st.button("Classify", type="primary", use_container_width=True, disabled=not model_exists)

    if run_btn:
        if not img_path:
            st.error("Please enter an image path.")
            return
        p = Path(img_path)
        if not p.exists():
            st.error(f"File not found: {img_path}")
            return
        _run_single_inference(p)

    st.divider()
    st.caption(
        "Confidence band: predictions below 60% confidence are labeled "
        '"uncertain" rather than forcing a binary guess. '
        "AI detection is never 100% certain."
    )

    # ---- Grad-CAM explainability ----
    st.divider()
    st.subheader("Explain with Grad-CAM")
    st.caption(
        "Grad-CAM produces a class-activation heatmap showing which image "
        "regions influenced the model's prediction. Heatmaps are computed "
        "from real gradients — they are never fabricated."
    )
    explain_path = st.text_input(
        "Image path for Grad-CAM",
        placeholder="e.g. C:\\Users\\Joseph\\Pictures\\photo.jpg",
        help="Full path to a single image file.",
    )
    col_exp, col_exp_info = st.columns([1, 4])
    with col_exp:
        explain_btn = st.button(
            "Generate heatmap",
            type="secondary",
            use_container_width=True,
            disabled=not model_exists,
        )

    if explain_btn:
        if not explain_path:
            st.error("Please enter an image path.")
        else:
            ep = Path(explain_path)
            if not ep.exists():
                st.error(f"File not found: {explain_path}")
            else:
                _run_gradcam(ep)


def _run_single_inference(path: Path) -> None:
    """Run AI inference on one image and display the result."""
    classifier = AIClassifier(
        model_path=Path(settings.ai_model_path),
        image_size=settings.ai_image_size,
        uncertainty_threshold=0.60,
    )

    with st.spinner("Classifying..."):
        try:
            result = classifier.predict_path(path)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

    # Display result
    col1, col2 = st.columns([1, 2])
    with col1:
        preview = load_preview_image(path, max_size=300)
        if preview:
            st.image(preview, caption=path.name, use_container_width=True)
        else:
            st.caption("(preview unavailable)")

    with col2:
        st.markdown(f"**File:** `{path.name}`")
        st.markdown(f"**Path:** `{path}`")

        label = result.label
        kind = {
            "authentic": "authentic",
            "ai_generated": "ai",
            "uncertain": "uncertain",
        }.get(label, "info")

        st.markdown("**Classification:** ", end="")
        status_badge(label.replace("_", " ").title(), kind)

        st.markdown(f"**Confidence:** {result.confidence:.1%}")
        st.markdown(f"**AI probability:** {result.ai_probability:.1%}")

        if result.uncertain:
            st.info("Result is uncertain — confidence below threshold.")
        if result.error:
            st.error(result.error)

    # Save to database
    try:
        engine = __import__("sentinel.app.ui.common", fromlist=["_get_engine"])._get_engine()
        with get_session(engine) as session:
            image = repo.get_image_by_path(session, str(path))
            if image is None:
                from sentinel.app.core.image_loader import get_basic_info
                info = get_basic_info(path)
                image = repo.create_image(
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
                        "scanned_at": __import__("datetime").datetime.now(__import__("datetime").UTC),
                    },
                )
            repo.set_ai_result(
                session,
                image=image,
                label=result.label,
                confidence=result.confidence,
                model_version="lightweight_cnn_v1",
                explanation=(
                    f"AI probability: {result.ai_probability:.3f}; "
                    f"uncertain: {result.uncertain}"
                ),
            )
            session.commit()
        st.success("Result saved to database.")
    except Exception as exc:
        st.warning(f"Could not save to database: {exc}")


def _run_gradcam(path: Path) -> None:
    """Run Grad-CAM explainability on one image and display heatmap."""
    explainer = GradCAMExplainer(
        model_path=Path(settings.ai_model_path),
        image_size=settings.ai_image_size,
    )

    with st.spinner("Generating Grad-CAM heatmap..."):
        try:
            result = explainer.explain_path(path)
        except Exception as exc:
            st.error(f"Grad-CAM failed: {exc}")
            return

    if result.error:
        st.error(result.error)
        return

    # Display original + heatmap side by side
    col1, col2 = st.columns(2)
    with col1:
        preview = load_preview_image(path, max_size=300)
        if preview:
            st.image(preview, caption=f"Original: {path.name}", use_container_width=True)
        else:
            st.caption("(preview unavailable)")
    with col2:
        if result.overlay is not None:
            # Convert BGR to RGB for Streamlit display
            import cv2
            overlay_rgb = cv2.cvtColor(result.overlay, cv2.COLOR_BGR2RGB)
            st.image(
                overlay_rgb,
                caption=f"Grad-CAM heatmap (α=0.5)",
                use_container_width=True,
            )
        else:
            st.caption("(heatmap unavailable)")

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Classification", result.label.replace("_", " ").title())
    c2.metric("AI probability", f"{result.ai_probability:.1%}")
    c3.metric("Confidence", f"{result.confidence:.1%}")

    st.caption(
        "Red/yellow regions contributed most to the prediction; blue regions "
        "contributed least. Heatmap is a gradient-based explanation, not "
        "definitive proof of AI generation."
    )

    explainer.close()

