from pathlib import Path
import logging
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_pdfs():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    markdown_files = []

    for pdf in RAW_DIR.glob("*.pdf"):

        md_file = PROCESSED_DIR / f"{pdf.stem}.md"

        # Skip conversion if Markdown already exists
        if md_file.exists() and md_file.stat().st_size > 0:
            print(f"Already exists: {md_file.name}")
            markdown_files.append(md_file)
            continue

        # Convert PDF to Markdown
        result = converter.convert(str(pdf))
        markdown = result.document.export_to_markdown()

        if not markdown.strip():
            raise ValueError(f"No text extracted from {pdf.name}")

        md_file.write_text(markdown, encoding="utf-8")

        print(f"Converted: {pdf.name} -> {md_file.name}")
        markdown_files.append(md_file)

    if not markdown_files:
        raise FileNotFoundError("No PDF or existing Markdown files found.")

    return markdown_files
