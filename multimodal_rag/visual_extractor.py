"""
Docling Visual Extractor for Multimodal RAG Ingestion.

Parses raw PDF documents with Docling to detect and extract tables, charts, and diagrams,
saving cropped high-resolution PNG images and Markdown representations alongside page
provenance metadata.
"""


import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem
from PIL import Image

from pathlib import Path

RAW_DIR = Path("data/raw")
IMAGE_DIR = Path("data/processed/extracted_images")


logger = logging.getLogger(__name__)


@dataclass
class ExtractedVisual:
    source: str
    page: int
    content_type: str
    visual_index: int
    image_path: Optional[str] = None
    table_markdown: Optional[str] = None
    caption: Optional[str] = None
    headings: Optional[list[str]] = None


class VisualExtractor:
    def __init__(self, images_scale: float = 2.0):
        self.images_scale = images_scale
        pipeline_options = PdfPipelineOptions(
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=images_scale,
            do_table_structure=True,
        )
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    def _save_image(self, image: Image.Image, output_path: Path) -> Optional[str]:
        if image is None:
            logger.warning("No image available for %s", output_path.name)
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        image.save(output_path, format="PNG")
        logger.info("Saved image: %s", output_path)
        return str(output_path.resolve())

    def _get_page_number(self, item):
        if not item.prov:
            return None

        return item.prov[0].page_no

    def _extract_table(self, item: TableItem, doc, source: str, output_dir: Path, index: int) -> Optional[ExtractedVisual]:
        page_number = self._get_page_number(item)
        if page_number is None:
            logger.warning("Table %s has no page provenance.", index)
            return None

        table_markdown = item.export_to_markdown(doc=doc)
        if not table_markdown or not table_markdown.strip():
            logger.warning("Empty table on page %s in %s", page_number, source)
            return None

        image_path = self._save_image(item.get_image(doc), output_dir / f"table_{index:03d}_page_{page_number}.png")
        if image_path is None:
            return None

        return ExtractedVisual(
            source=source,
            page=page_number,
            content_type="table",
            visual_index=index,
            image_path=image_path,
            table_markdown=table_markdown,
            caption=None
        )

    def _extract_picture(self, item: PictureItem, doc, source: str, output_dir: Path, index: int) -> Optional[ExtractedVisual]:
        page_number = self._get_page_number(item)
        if page_number is None:
            logger.warning("Picture %s has no page provenance.", index)
            return None

        image_path = self._save_image(item.get_image(doc), output_dir / f"picture_{index:03d}_page_{page_number}.png")
        if image_path is None:
            return None

        try:
            caption = item.caption_text(doc)
        except Exception:
            caption = None
            logger.debug("No picture caption available on page %s.", page_number)

        return ExtractedVisual(
            source=source,
            page=page_number,
            content_type="picture",
            visual_index=index,
            image_path=image_path,
            table_markdown=None,
            caption=caption
        )

    def extract_pdf(self, pdf_path: Path) -> list[ExtractedVisual]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file: {pdf_path}")

        logger.info("Processing visuals from: %s", pdf_path.name)
        doc = self.converter.convert(str(pdf_path)).document
        output_dir = IMAGE_DIR / pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted_visuals = []
        table_index = picture_index = 0

        current_headings: list[str] = []

        for item, _ in doc.iterate_items():
            # Track section headers as Docling walks down the page
            label = getattr(getattr(item, "label", None), "name", "")
            if label in ("SECTION_HEADER", "TITLE") and getattr(item, "text", "").strip():
                current_headings = [item.text.strip()]

            if isinstance(item, TableItem):
                table_index += 1
                visual = self._extract_table(item, doc, pdf_path.name, output_dir, table_index)
                if visual:
                    visual.headings = list(current_headings)
                    extracted_visuals.append(visual)
            elif isinstance(item, PictureItem):
                picture_index += 1
                visual = self._extract_picture(item, doc, pdf_path.name, output_dir, picture_index)
                if visual:
                    visual.headings = list(current_headings)
                    extracted_visuals.append(visual)

        
        logger.info("Finished %s: %d tables, %d pictures", pdf_path.name, table_index, picture_index)
        return extracted_visuals

    def extract_all(self) -> list[ExtractedVisual]:
        if not RAW_DIR.exists():
            raise FileNotFoundError(f"Raw PDF directory missing: {RAW_DIR}")

        pdf_files = sorted(RAW_DIR.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {RAW_DIR}")

        all_visuals = []
        for pdf_path in pdf_files:
            all_visuals.extend(self.extract_pdf(pdf_path))

        logger.info("Total extracted visual elements: %d", len(all_visuals))
        return all_visuals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    extractor = VisualExtractor(images_scale=2.0)
    visuals = extractor.extract_all()

    logger.info("Extraction summary: %d visuals", len(visuals))
    for visual in visuals:
        logger.info(
            "Source: %s | Page: %d | Type: %s | Image: %s",
            visual.source,
            visual.page,
            visual.content_type,
            visual.image_path,
        )
        if visual.table_markdown:
            logger.info("Table Markdown:\n%s", visual.table_markdown[:500])
