"""Resume document parsing utilities.

The parser supports text-based PDF and DOCX files and rejects documents that
do not contain enough readable text for reliable AI analysis.
"""

import io
from pypdf import PdfReader
import docx


def extract_text_from_pdf(file) -> str:
    """Extracts raw text from a PDF file page by page."""
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file) -> str:
    """Extracts raw text from a Word document paragraph by paragraph."""
    text = ""
    doc = docx.Document(file)
    for para in doc.paragraphs:
        if para.text:
            text += para.text + "\n"
    return text.strip()

    
def parse_resume(uploaded_file) -> str:
    """
    Supervisor function:
    1. Inspects the file type (.pdf or .docx)
    2. Extracts digital text
    3. Defensive check: verifies the resume isn't a blank or scanned image
    """
    filename = uploaded_file.name.lower()
    extracted_text = ""

    if filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        extracted_text = extract_text_from_docx(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a .pdf or .docx file.")

    # 🛡️ Defensive Guardrail:
    # If the text has fewer than 50 characters, it's likely a blank file or a scanned image
    if len(extracted_text.strip()) < 50:
        raise ValueError(
            "We could not detect sufficient readable text in your resume. "
            "If your resume is a scanned image or photo, please upload a text-searchable PDF or Word document."
        )

    return extracted_text