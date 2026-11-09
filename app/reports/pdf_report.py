"""PDF report generation for Sentinel.

Uses ReportLab (already installed) to produce a self-contained PDF scan
report containing the summary, exact duplicate groups, near-duplicate pairs,
and AI detection results.  This is an optional extension to the CSV/JSON
reports already provided by reports_page.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

from sentinel.app.config import REPORTS_DIR


def _styles() -> dict:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#16213e"),
            spaceBefore=14, spaceAfter=6,
        ),
        "body": styles["BodyText"],
        "small": ParagraphStyle(
            "Small", parent=styles["BodyText"], fontSize=8, leading=10,
            textColor=colors.HexColor("#4a4a4a"),
        ),
    }


def _header_footer(canvas, doc):
    """Draw a simple header/footer on each page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.75 * inch, 0.5 * inch, "SENTINEL — Scan Report")
    canvas.drawRightString(
        letter[0] - 0.75 * inch, 0.5 * inch,
        f"Page {doc.page} | Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    )
    canvas.restoreState()


def generate_pdf_report(report: dict) -> bytes:
    """Generate a PDF report from a report dictionary (same shape as the JSON report).

    Parameters
    ----------
    report : dict
        Output of `_build_report` from reports_page.py — contains
        ``summary``, ``exact_duplicate_groups``, ``near_duplicate_pairs``,
        ``ai_detection_results``, and ``generated_at``.

    Returns
    -------
    bytes
        PDF document bytes.
    """
    styles = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.85 * inch,
        title="SENTINEL Scan Report",
        author="Sentinel — Image Authenticity & Duplicate Analyzer",
    )

    story: list = []

    # ---- Title block ----
    story.append(Paragraph("SENTINEL — Scan Report", styles["title"]))
    story.append(
        Paragraph(
            f"Generated: {report.get('generated_at', datetime.now(timezone.utc).isoformat())}",
            styles["small"],
        )
    )
    story.append(Spacer(1, 8))

    # ---- Summary section ----
    story.append(Paragraph("1. Scan Summary", styles["h2"]))
    summary = report.get("summary", {})
    summary_rows = [
        ["Metric", "Value"],
        ["Total images scanned", str(summary.get("total_images", 0))],
        ["Exact duplicate groups", str(summary.get("exact_duplicate_groups", 0))],
        ["Near-duplicate pairs", str(summary.get("near_duplicate_pairs", 0))],
        ["Potential savings (bytes)", str(summary.get("potential_savings_bytes", 0))],
        ["AI-generated count", str(summary.get("ai_generated_count", 0))],
        ["AI authentic count", str(summary.get("ai_authentic_count", 0))],
        ["AI inconclusive count", str(summary.get("ai_uncertain_count", 0))],
    ]
    summary_table = Table(summary_rows, colWidths=[2.6 * inch, 3.4 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # ---- Exact duplicates ----
    story.append(Paragraph("2. Exact Duplicate Groups", styles["h2"]))
    exact_groups = report.get("exact_duplicate_groups", [])
    if not exact_groups:
        story.append(Paragraph("No exact duplicate groups found.", styles["body"]))
    else:
        exact_rows = [
            ["Group ID", "SHA-256 (prefix)", "Files", "Total (bytes)", "Redundant (bytes)"]
        ]
        for g in exact_groups:
            exact_rows.append([
                str(g["group_id"]),
                (g.get("sha256") or "")[:16] + "...",
                str(g["file_count"]),
                str(g["total_size_bytes"]),
                str(g["redundant_size_bytes"]),
            ])
        exact_table = Table(
            exact_rows,
            colWidths=[0.7 * inch, 1.6 * inch, 0.5 * inch, 1.0 * inch, 1.2 * inch],
            repeatRows=1,
        )
        exact_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(exact_table)
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "Total redundant storage: "
                f"{sum(g.get('redundant_size_bytes', 0) for g in exact_groups):,} bytes",
                styles["small"],
            )
        )

    # ---- Near duplicates ----
    story.append(PageBreak())
    story.append(Paragraph("3. Near-Duplicate Pairs", styles["h2"]))
    near_pairs = report.get("near_duplicate_pairs", [])
    if not near_pairs:
        story.append(Paragraph("No near-duplicate pairs found.", styles["body"]))
    else:
        near_rows = [
            ["Pair ID", "Method", "Similarity", "Hamming", "Image A", "Image B"]
        ]
        for p in near_pairs[:50]:  # cap rows for readability
            near_rows.append([
                str(p["edge_id"]),
                p.get("method", "—"),
                f"{p.get('similarity_score', 0):.0%}",
                str(p.get("hamming_distance") or "—"),
                p["image_a"]["file_name"][:28],
                p["image_b"]["file_name"][:28],
            ])
        near_table = Table(
            near_rows,
            colWidths=[0.55 * inch, 0.75 * inch, 0.85 * inch, 0.6 * inch, 1.5 * inch, 1.5 * inch],
            repeatRows=1,
        )
        near_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(near_table)
        if len(near_pairs) > 50:
            story.append(Paragraph(
                f"Showing first 50 of {len(near_pairs)} pairs.", styles["small"]
            ))

    # ---- AI detection ----
    story.append(Spacer(1, 12))
    story.append(Paragraph("4. AI Detection Results", styles["h2"]))
    ai_results = report.get("ai_detection_results", [])
    if not ai_results:
        story.append(Paragraph("No AI detection results available.", styles["body"]))
    else:
        ai_rows = [
            ["Image", "Label", "Confidence", "Model version"]
        ]
        for r in ai_results[:50]:
            ai_rows.append([
                r["file_name"][:34],
                r.get("label", "—").replace("_", " ").title(),
                f"{r.get('confidence', 0):.0%}",
                r.get("model_version") or "—",
            ])
        ai_table = Table(
            ai_rows,
            colWidths=[2.2 * inch, 1.5 * inch, 0.9 * inch, 1.4 * inch],
            repeatRows=1,
        )
        ai_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(ai_table)
        if len(ai_results) > 50:
            story.append(Paragraph(
                f"Showing first 50 of {len(ai_results)} results.", styles["small"]
            ))

    # ---- Footer note ----
    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Sentinel — AI Authenticity & Intelligent Duplicate Image Analyzer. "
        "All processing performed locally. AI detection results are probabilistic "
        "and should be reviewed manually; they are not 100% certain.",
        styles["small"],
    ))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


def save_pdf_report(report: dict, output_path: Optional[str] = None) -> str:
    """Generate a PDF report and save it to disk.

    Returns the path to the saved PDF file.
    """
    from pathlib import Path
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = str(REPORTS_DIR / f"sentinel_report_{ts}.pdf")
    pdf_bytes = generate_pdf_report(report)
    Path(output_path).write_bytes(pdf_bytes)
    return output_path
