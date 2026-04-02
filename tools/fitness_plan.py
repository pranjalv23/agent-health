import logging
import os
import re
import uuid
from datetime import datetime, timezone

from langchain_core.tools import tool

logger = logging.getLogger("agent_health.tools.fitness_plan")

_BASE_URL = (os.getenv("BACKEND_URL") or os.getenv("PUBLIC_URL") or "").rstrip("/")

_UNICODE_TO_ASCII = str.maketrans({
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "≤": "<=", "≥": ">=", "≠": "!=", "→": "->",
    "·": ".", "•": "*",
})


def _sanitize_for_pdf(text: str) -> str:
    return text.translate(_UNICODE_TO_ASCII)


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:50]


def _create_pdf_bytes(title: str, markdown_content: str) -> bytes:
    """Generate a PDF from markdown content using fpdf2. Returns bytes."""
    from fpdf import FPDF

    markdown_content = _sanitize_for_pdf(markdown_content)
    title = _sanitize_for_pdf(title)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Content — line-by-line markdown rendering
    pdf.set_font("Helvetica", size=11)
    for line in markdown_content.split("\n"):
        stripped = line.strip()

        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.ln(4)
            pdf.cell(0, 10, stripped[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(3)
            pdf.cell(0, 9, stripped[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.ln(2)
            pdf.cell(0, 8, stripped[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            bullet_text = stripped[2:]
            bullet_text = re.sub(r"\*\*(.*?)\*\*", r"\1", bullet_text)
            pdf.cell(8)
            pdf.multi_cell(0, 6, f"• {bullet_text}")
        elif stripped:
            plain = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
            plain = re.sub(r"\*(.*?)\*", r"\1", plain)
            pdf.multi_cell(0, 6, plain)
        else:
            pdf.ln(3)

    return pdf.output()


@tool
async def generate_fitness_plan(title: str, content: str, format: str = "pdf") -> str:
    """Generate a downloadable fitness plan document from the provided content.

    Args:
        title: The plan title (e.g. "4-Week Beginner Workout Plan", "High-Protein Meal Plan").
        content: Full markdown content of the plan. Use ## for week/phase headers,
                 ### for day/section headers, and bullet points for exercises/meals.
                 Include sets, reps, rest periods, macro targets, and timing guidance.
        format: Output format — "pdf" or "markdown". Defaults to "pdf".
    """
    from database.mongo import MongoDB

    file_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _slugify(title)

    if format == "pdf":
        filename = f"{timestamp}_{slug}.pdf"
    else:
        filename = f"{timestamp}_{slug}.md"

    try:
        if format == "pdf":
            file_bytes = _create_pdf_bytes(title, content)
        else:
            full_content = f"# {title}\n\n{content}"
            file_bytes = full_content.encode("utf-8")

        await MongoDB.store_file(
            file_id=file_id,
            filename=filename,
            data=file_bytes,
            file_type="fitness_plan",
        )

        logger.info("Generated fitness plan: file_id='%s', format='%s', size=%d bytes",
                     file_id, format, len(file_bytes))

        return (
            f"Fitness plan generated successfully!\n\n"
            f"**Title:** {title}\n"
            f"**Format:** {format.upper()}\n"
            f"**Download:** [Download: {title}]({_BASE_URL}/download/{file_id})"
        )

    except Exception as e:
        logger.error("Failed to generate fitness plan: %s", e)
        return (
            f"Error generating fitness plan ({format}): {e}. "
            "Do NOT retry with the same format. "
            "If format was 'pdf', call this tool again with format='markdown' instead."
        )
