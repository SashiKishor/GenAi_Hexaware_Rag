
import logging
import re
from pathlib import Path

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def extract_text(markdown_files: list[Path]) -> list[Document]:

    documents = []

    for md_path in markdown_files:

        md_path = Path(md_path)

        if not md_path.exists():
            raise FileNotFoundError(
                f"Markdown not found: {md_path}"
            )

        text = md_path.read_text(encoding="utf-8")

        if not text.strip():
            logger.warning(
                "Empty Markdown file: %s",
                md_path.name
            )
            continue

        year_match = re.search(
            r"20\d{2}",
            md_path.stem
        )

        year = year_match.group() if year_match else "unknown"

        # Split Markdown using the page markers
        pages = re.split(
            r"<!-- PAGE_NUMBER: (\d+) -->",
            text
        )

        # Page markers produce alternating text and page numbers
        for index in range(1, len(pages), 2):

            page_number = int(pages[index])
            page_content = pages[index + 1].strip()

            if not page_content:
                continue

            document = Document(
                page_content=page_content,
                metadata={
                    "source": md_path.stem + ".pdf",
                    "version": year,
                    "document_year": year,
                    "page": page_number,
                    "document_type": "bank_document",
                    "file_type": "pdf",
                    "status": (
                        "current" if year == "2026"
                        else "superseded" if year == "2025"
                        else "unknown"
                    )
                }
            )

            documents.append(document)

        logger.info(
            "Extracted pages from: %s",
            md_path.name
        )

    if not documents:
        raise ValueError("No usable text extracted.")

    return documents
