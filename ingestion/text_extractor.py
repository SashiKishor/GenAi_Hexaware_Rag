
import logging
import re
from pathlib import Path

from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text(markdown_files: list[Path]) -> list[Document]:

    documents = []

    for md_path in markdown_files:

        md_path = Path(md_path)

        if not md_path.exists():
            raise FileNotFoundError(f"Markdown not found: {md_path}")

        text = md_path.read_text(encoding="utf-8")

        if not text.strip():
            logger.warning("Empty Markdown file: %s", md_path.name)
            continue

        year_match = re.search(r"20\d{2}", md_path.stem)

        year = year_match.group() if year_match else "unknown"

        document = Document(
            page_content=text,
            metadata={
                "source": md_path.name,
                "document_year": year,
                "file_type": "markdown"
            }
        )

        documents.append(document)

        logger.info("Extracted text: %s", md_path.name)

    if not documents:
        raise ValueError("No usable text extracted.")

    return documents
