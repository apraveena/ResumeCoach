import io
import re
import docx
from docx.shared import Inches


def modify_existing_docx(uploaded_file, star_rewrites: list) -> io.BytesIO:
    """
    PATH 1: In-place replacement for uploaded .docx files.
    Preserves 100% of original fonts, margins, headers, and colors.
    """
    uploaded_file.seek(0)
    doc = docx.Document(uploaded_file)

    for item in star_rewrites:
        original = item.get("original", "").strip()
        rewritten = item.get("rewritten", "").strip()

        if not original or not rewritten:
            continue

        # 1. Search and replace inside standard paragraphs
        for paragraph in doc.paragraphs:
            if original in paragraph.text:
                paragraph.text = paragraph.text.replace(original, rewritten)

        # 2. Search and replace inside any tables (e.g., two-column resume layouts)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        if original in p.text:
                            p.text = p.text.replace(original, rewritten)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def create_clean_ats_docx(original_text: str, star_rewrites: list) -> io.BytesIO:
    """
    PATH 2: Creates an ATS-standard Word document from extracted text (for PDF uploads).
    """
    tailored_text = original_text
    for item in star_rewrites:
        original = item.get("original", "").strip()
        rewritten = item.get("rewritten", "").strip()
        if original and rewritten:
            tailored_text = tailored_text.replace(original, rewritten)

    doc = docx.Document()

    # Standard ATS 1-inch margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    lines = tailored_text.split("\n")
    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue

        # Bullet symbols (-, *, •)
        if cleaned_line.startswith(("-", "*", "•")):
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(cleaned_line.lstrip("-*• ").strip())
        # Numbered lists (1., 2., 1))
        elif re.match(r"^\d+[\.\)]\s+", cleaned_line):
            clean_text = re.sub(r"^\d+[\.\)]\s+", "", cleaned_line)
            p = doc.add_paragraph(style="List Number")
            p.add_run(clean_text.strip())
        else:
            doc.add_paragraph(cleaned_line)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def generate_tailored_resume(uploaded_file, original_text: str, star_rewrites: list) -> io.BytesIO:
    """
    Supervisor Dispatcher:
    - If user uploaded a .docx -> do in-place styling preservation
    - If user uploaded a .pdf  -> generate an ATS-standard clean .docx
    """
    filename = uploaded_file.name.lower()
    if filename.endswith(".docx"):
        return modify_existing_docx(uploaded_file, star_rewrites)
    else:
        return create_clean_ats_docx(original_text, star_rewrites)


def generate_text_coaching_report(results: dict) -> str:
    """
    Generates a clean plain-text coaching summary report (.txt).
    """
    report = []
    report.append("=" * 60)
    report.append("          🎯 RESUME COACH AI - EVALUATION REPORT")
    report.append("=" * 60)
    report.append(f"\nATS MATCH SCORE: {results.get('match_score', 0)}%\n")
    report.append("EXECUTIVE SUMMARY:")
    report.append(results.get("summary", "N/A"))
    report.append("\n" + "-" * 60)
    report.append("MATCHING STRENGTHS:")
    for skill in results.get("matching_skills", []):
        report.append(f"  • {skill}")

    report.append("\nMISSING CRITICAL SKILLS:")
    for skill in results.get("missing_skills", []):
        report.append(f"  • [GAP] {skill}")

    report.append("\n" + "-" * 60)
    report.append(f"STAR BULLET REWRITES ({results.get('rewrites_summary', '')}):\n")
    for idx, item in enumerate(results.get("star_rewrites", []), 1):
        report.append(f"[{idx}] ORIGINAL:")
        report.append(f"    {item.get('original')}")
        report.append(f"    OPTIMIZED (STAR):")
        report.append(f"    {item.get('rewritten')}")
        report.append(f"    WHY:")
        report.append(f"    {item.get('reason')}\n")

    report.append("=" * 60)
    return "\n".join(report)