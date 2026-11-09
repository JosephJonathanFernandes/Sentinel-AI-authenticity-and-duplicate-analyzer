"""Reports page — CSV/JSON export and scan summaries."""

from __future__ import annotations

from pathlib import Path
import csv
import io
import json
from datetime import datetime, timezone

import streamlit as st

from sentinel.app.config import REPORTS_DIR
from sentinel.app.database.database import get_session
from sentinel.app.database import repository as repo
from sentinel.app.reports.pdf_report import generate_pdf_report
from .common import format_size, init_session_state


def render_reports_page() -> None:
    st.title("Reports")
    st.markdown(
        "Export scan results as CSV or JSON. Reports include duplicate groups, "
        "near-duplicate pairs, AI detection results, and forensic summaries."
    )

    engine = __import__("sentinel.app.ui.common", fromlist=["_get_engine"])._get_engine()

    from sqlalchemy import select

    from sentinel.app.database.models import Image

    with get_session(engine) as session:
        total_images = session.execute(select(Image)).scalars().count()
        exact_groups = repo.get_exact_groups(session)
        near_edges = repo.get_near_duplicate_edges(session, limit=10000)
        ai_results = repo.get_ai_results(session, limit=10000)

    st.metric("Images in database", total_images)
    st.metric("Exact duplicate groups", len(exact_groups))
    st.metric("Near-duplicate pairs", len(near_edges))
    st.metric("AI results", len(ai_results))

    savings = sum(g.redundant_size_bytes for g in exact_groups)
    st.metric("Potential storage savings", format_size(savings))

    st.divider()

    # Build report data
    report = _build_report(session, exact_groups, near_edges, ai_results, savings)

    col_csv, col_json, col_pdf = st.columns(3)
    with col_csv:
        csv_bytes = _report_to_csv(report)
        st.download_button(
            "Download CSV report",
            data=csv_bytes,
            file_name=f"sentinel_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_json:
        json_bytes = _report_to_json(report)
        st.download_button(
            "Download JSON report",
            data=json_bytes,
            file_name=f"sentinel_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_pdf:
        try:
            pdf_bytes = generate_pdf_report(report)
            st.download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name=f"sentinel_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.button(
                "PDF report unavailable",
                disabled=True,
                use_container_width=True,
                help=str(exc),
            )

    st.divider()
    st.subheader("Report preview (JSON)")
    st.json(report, expanded=False)


def _build_report(session, exact_groups, near_edges, ai_results, savings) -> dict:
    """Assemble a report dictionary from database contents."""
    exact_group_list = []
    for g in exact_groups:
        members = []
        for m in g.members:
            members.append(
                {
                    "image_id": m.image_id,
                    "file_name": m.image.file_name,
                    "path": m.image.path,
                    "size_bytes": m.image.size_bytes,
                    "width": m.image.width,
                    "height": m.image.height,
                    "format": m.image.format,
                    "sha256": m.image.sha256,
                    "is_primary": m.is_primary,
                }
            )
        exact_group_list.append(
            {
                "group_id": g.id,
                "sha256": g.sha256,
                "file_count": g.file_count,
                "total_size_bytes": g.total_size_bytes,
                "redundant_size_bytes": g.redundant_size_bytes,
                "members": members,
            }
        )

    near_dup_list = []
    for e in near_edges:
        near_dup_list.append(
            {
                "edge_id": e.id,
                "image_a": {
                    "image_id": e.image_a.id,
                    "file_name": e.image_a.file_name,
                    "path": e.image_a.path,
                },
                "image_b": {
                    "image_id": e.image_b.id,
                    "file_name": e.image_b.file_name,
                    "path": e.image_b.path,
                },
                "similarity_score": e.similarity_score,
                "hamming_distance": e.hamming_distance,
                "method": e.method,
            }
        )

    ai_list = []
    for r in ai_results:
        ai_list.append(
            {
                "image_id": r.image_id,
                "file_name": r.image.file_name,
                "path": r.image.path,
                "label": r.label,
                "confidence": r.confidence,
                "model_version": r.model_version,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_images": total_images,
            "exact_duplicate_groups": len(exact_group_list),
            "near_duplicate_pairs": len(near_dup_list),
            "potential_savings_bytes": savings,
            "ai_generated_count": sum(1 for r in ai_results if r.label == "ai_generated"),
            "ai_authentic_count": sum(1 for r in ai_results if r.label == "authentic"),
            "ai_uncertain_count": sum(1 for r in ai_results if r.label == "uncertain"),
        },
        "exact_duplicate_groups": exact_group_list,
        "near_duplicate_pairs": near_dup_list,
        "ai_detection_results": ai_list,
    }


def _report_to_csv(report: dict) -> bytes:
    """Flatten the report into CSV bytes."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["SENTINEL Scan Report"])
    writer.writerow(["Generated at", report["generated_at"]])
    writer.writerow([])

    writer.writerow(["Summary"])
    for k, v in report["summary"].items():
        writer.writerow([k, v])
    writer.writerow([])

    writer.writerow(["Exact Duplicate Groups"])
    writer.writerow(
        ["group_id", "sha256", "file_count", "total_size_bytes", "redundant_size_bytes"]
    )
    for g in report["exact_duplicate_groups"]:
        writer.writerow(
            [
                g["group_id"],
                g["sha256"],
                g["file_count"],
                g["total_size_bytes"],
                g["redundant_size_bytes"],
            ]
        )
    writer.writerow([])

    writer.writerow(["Exact Duplicate Members"])
    writer.writerow(
        ["group_id", "image_id", "file_name", "path", "size_bytes", "width", "height", "format", "sha256", "is_primary"]
    )
    for g in report["exact_duplicate_groups"]:
        for m in g["members"]:
            writer.writerow(
                [
                    g["group_id"],
                    m["image_id"],
                    m["file_name"],
                    m["path"],
                    m["size_bytes"],
                    m["width"],
                    m["height"],
                    m["format"],
                    m["sha256"],
                    m["is_primary"],
                ]
            )
    writer.writerow([])

    writer.writerow(["Near Duplicate Pairs"])
    writer.writerow(
        ["edge_id", "image_a", "image_b", "similarity_score", "hamming_distance", "method"]
    )
    for p in report["near_duplicate_pairs"]:
        writer.writerow(
            [
                p["edge_id"],
                p["image_a"]["file_name"],
                p["image_b"]["file_name"],
                p["similarity_score"],
                p["hamming_distance"],
                p["method"],
            ]
        )
    writer.writerow([])

    writer.writerow(["AI Detection Results"])
    writer.writerow(["image_id", "file_name", "path", "label", "confidence", "model_version"])
    for r in report["ai_detection_results"]:
        writer.writerow(
            [r["image_id"], r["file_name"], r["path"], r["label"], r["confidence"], r["model_version"]]
        )

    return output.getvalue().encode("utf-8")


def _report_to_json(report: dict) -> bytes:
    return json.dumps(report, indent=2, default=str).encode("utf-8")
