
import logging
import re
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def load_pdfs():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {RAW_DIR}"
        )

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {RAW_DIR}"
        )

    markdown_files = []

    for pdf_path in pdf_files:

        logger.info("Reading PDF: %s", pdf_path.name)

        reader = PdfReader(str(pdf_path))

        markdown_content = []

        for page_number, page in enumerate(
            reader.pages, start=1
        ):

            page_text = page.extract_text() or ""

            markdown_content.append(
                f"<!-- PAGE_NUMBER: {page_number} -->\n"
                f"{page_text}\n"
            )

        markdown = "\n".join(markdown_content)

        if not markdown.strip():
            raise ValueError(
                f"No text extracted from {pdf_path.name}"
            )

        md_file = PROCESSED_DIR / f"{pdf_path.stem}.md"

        # Overwrite Markdown so changes to the PDF are reflected
        md_file.write_text(
            markdown,
            encoding="utf-8"
        )

        logger.info(
            "Converted %s into %s",
            pdf_path.name,
            md_file.name
        )

        markdown_files.append(md_file)

    return markdown_files
