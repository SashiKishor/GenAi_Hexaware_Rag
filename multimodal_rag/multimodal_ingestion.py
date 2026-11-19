"""
Unified Multimodal Ingestion Pipeline for Northbridge Bank.

Combines Docling hybrid text/table chunking with cropped visual extraction and Gemini
vision captioning, tagging every chunk with version status and effective dates before
storing the unified corpus in ChromaDB.
"""


import json
import logging
from pathlib import Path

from vector_store.chroma_db import ChromaDB
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter

from multimodal_rag.multimodal_chunking import (
    chunk_docling_document,
    print_chunk_statistics
)

from multimodal_rag.models import (
    build_metadata,
    create_document,
    determine_document_status,
    extract_document_year,
    extract_effective_date,
)
from multimodal_rag.visual_captioner import VisualCaptioner
from multimodal_rag.visual_extractor import VisualExtractor

RAW_DIR = Path("data/raw")
IMAGE_DIR = Path("data/processed/extracted_images")

logger = logging.getLogger(__name__)

class MultimodalIngestion:
    def __init__(self, visual_extractor=None, visual_captioner=None, chroma_db=None):
        self.visual_extractor = visual_extractor or VisualExtractor()
        self.visual_captioner = visual_captioner or VisualCaptioner()
        self.chroma_db = chroma_db or ChromaDB()

    def get_latest_year(self) -> str:
        years = [int(y) for pdf in RAW_DIR.glob("*.pdf") if (y := extract_document_year(pdf.name)).isdigit()]
        if not years:
            raise ValueError("No document versions found in PDF filenames.")
        return str(max(years))

    
    def ingest_text(self,latest_year: str) -> list[Document]:
        logger.info("Starting Docling text and table ingestion...")

        pdf_files = sorted(RAW_DIR.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError( f"No PDFs found in {RAW_DIR}")

        converter = DocumentConverter()
        all_chunks = []

        for pdf_path in pdf_files:
            logger.info("Converting PDF using Docling: %s",pdf_path.name)
            result = converter.convert(str(pdf_path)) 
            if result.document is None:
                logger.error("Docling returned no document for %s",pdf_path.name)
                continue

            chunks = chunk_docling_document(
                dl_doc=result.document,
                source=pdf_path.name,
                latest_year=latest_year,
            )
            
            # Keep only text chunks here; ingest_visuals() handles all tables with PNGs & numeric_facts
            text_only = [c for c in chunks if c.metadata.get("content_type") != "table"]
            for c in text_only:
                headings = json.loads(c.metadata.get("headings", "[]"))
                c.metadata["section"] = " > ".join(headings) if headings else ""

            all_chunks.extend(text_only)

        if not all_chunks:
            raise ValueError("Docling did not generate any usable chunks.")
            
        print_chunk_statistics(all_chunks)

        return all_chunks

    def build_visual_content(self, visual, analysis: dict | None) -> str:
        parts = []
        if visual.table_markdown:
            parts.append(f"TABLE CONTENT:\n{visual.table_markdown}")
        if visual.caption:
            parts.append(f"DOCUMENT CAPTION:\n{visual.caption}")

        if analysis:
            if caption := analysis.get("caption"):
                parts.append(f"VISUAL DESCRIPTION:\n{caption}")
            if transcribed := analysis.get("transcribed_text"):
                parts.append(f"TRANSCRIBED TEXT:\n{transcribed}")
            if facts := analysis.get("numeric_facts"):
                fact_lines = [
                    f"{f.get('label', '')}: {f.get('value', '')} {f.get('unit', '')}. Condition: {f.get('condition', '')}"
                    for f in facts
                ]
                parts.append("NUMERIC FACTS:\n" + "\n".join(fact_lines))
            if rels := analysis.get("relationships"):
                parts.append("RELATIONSHIPS:\n" + "\n".join(rels))
            if uncert := analysis.get("uncertainties"):
                parts.append("UNCERTAINTIES:\n" + "\n".join(uncert))

        return "\n\n".join(parts).strip()

    def _resolve_content_type(self, raw_type: str, detected_type: str) -> str:
        if raw_type == "table" or detected_type == "table":
            return "table"
        if detected_type in ("chart", "diagram"):
            return detected_type
        return "diagram"

    def ingest_visuals(self, latest_year: str) -> list[Document]:
        logger.info("Starting visual ingestion...")
        visuals = self.visual_extractor.extract_all()
        visual_documents = []

        for idx, visual in enumerate(visuals, start=1):
            logger.info("Processing visual %d/%d: %s page %s", idx, len(visuals), visual.source, visual.page)
            analysis = None

            if visual.image_path:
                try:
                    v_type = "table" if visual.content_type == "table" else "visual"
                    analysis = self.visual_captioner.caption_image(image_path=visual.image_path, visual_type=v_type)
                except Exception:
                    logger.exception("Gemini analysis failed for %s", visual.image_path)

            content = self.build_visual_content(visual, analysis)
            if not content:
                logger.warning("Skipping empty visual: %s", visual.image_path)
                continue

            # --- NEW: Attach the active section header to both text content and metadata ---
            section_title = " > ".join(visual.headings) if getattr(visual, "headings", None) else ""
            if section_title:
                content = f"SECTION: {section_title}\n\n{content}"

            detected_type = analysis.get("visual_type", "") if analysis else ""
            content_type = self._resolve_content_type(visual.content_type, detected_type)

            metadata = build_metadata(
                source=visual.source,
                page=visual.page,
                content_type=content_type,
                content=content,
                latest_year=latest_year,
                image_path=visual.image_path,
                effective_date=extract_effective_date(content),
                numeric_facts=analysis.get("numeric_facts", []) if analysis else [],
                visual_type=detected_type,
            )
            metadata["section"] = section_title  # <--- NEW: Save section header for citations
            visual_documents.append(create_document(content=content, metadata=metadata))

        logger.info("Visual documents generated: %d", len(visual_documents))
        return visual_documents

    def run(self, force: bool = False) -> list[Document]:
        # Check if ChromaDB already has documents stored
        existing_count = self.chroma_db.vector_store._collection.count()
        if existing_count > 0 and not force:
            logger.info(
                "ChromaDB already contains %d documents. Skipping multimodal ingestion.",
                existing_count,
            )
            return []

        logger.info("Starting Northbridge multimodal RAG ingestion")
        latest_year = self.get_latest_year()
        logger.info("Latest document version: %s", latest_year)

        all_documents = self.ingest_text(latest_year) + self.ingest_visuals(latest_year)
        if not all_documents:
            raise ValueError("No documents generated for ingestion.")

        self.chroma_db.add_chunks(all_documents)
        logger.info("Successfully stored %d documents.", len(all_documents))
        return all_documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    documents = MultimodalIngestion().run()

    logger.info("MULTIMODAL INGESTION SUMMARY | Total stored: %d", len(documents))
    for c_type in ("text", "table", "chart", "diagram"):
        count = sum(1 for d in documents if d.metadata.get("content_type") == c_type)
        logger.info("  %s: %d", c_type, count)

    if documents:
        logger.info("Sample document metadata: %s", documents[-1].metadata)
