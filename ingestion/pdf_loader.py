
from pathlib import Path
import logging

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_pdfs(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = PROCESSED_DIR
) -> list[Path]:

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found.")

    converter = DocumentConverter()
    markdown_files = []

    for pdf_path in pdf_files:

        md_path = output_dir / f"{pdf_path.stem}.md"

        # Reuse existing Markdown file
        if md_path.exists() and md_path.stat().st_size > 0:

            logger.info("Markdown already exists. Skipping: %s", md_path.name)
            markdown_files.append(md_path)
            continue

        logger.info("Converting PDF: %s", pdf_path.name)

        try:
            result = converter.convert(str(pdf_path))

            markdown_text = result.document.export_to_markdown()

            if not markdown_text.strip():
                raise ValueError("Extracted Markdown is empty.")

            md_path.write_text(
                markdown_text,
                encoding="utf-8"
            )

            markdown_files.append(md_path)

            logger.info("Created Markdown: %s", md_path.name)

        except Exception as exc:
            logger.exception("PDF conversion failed: %s", pdf_path.name)
            raise RuntimeError(
                f"Failed to convert {pdf_path.name}"
            ) from exc

    return markdown_files
