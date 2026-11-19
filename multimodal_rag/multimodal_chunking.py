"""
Docling Hybrid Chunker for Multimodal RAG Ingestion.

Splits a parsed DoclingDocument into structure-aware LangChain Document chunks,
contextualizing each chunk with section headings, page numbers, version status,
effective dates, and deterministic SHA-256 chunk IDs for ChromaDB storage.
"""


import hashlib
import json
import logging
import re
from datetime import datetime

from docling.chunking import HybridChunker
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def create_hybrid_chunker() -> HybridChunker:
    return HybridChunker(
        merge_peers=True,
        repeat_table_header=True,
        omit_header_on_overflow=False,
    )


def extract_year(source: str) -> str:
    match = re.search(r"20\d{2}", source)
    return match.group(0) if match else "unknown"


def get_document_status(year: str, latest_year: str) -> str:
    if year == "unknown" or latest_year == "unknown":
        return "unknown"
    if year == latest_year:
        return "current"
    if int(year) < int(latest_year):
        return "superseded"
    return "unknown"


def extract_page_number(chunk) -> int | None:
    origin = getattr(chunk.meta, "origin", None)
    if (page_no := getattr(origin, "page_no", None)) is not None:
        return int(page_no)

    for item in getattr(chunk.meta, "doc_items", []):
        provs = getattr(item, "prov", [])
        if not isinstance(provs, list):
            provs = [provs]
        for prov in provs:
            if (page_no := getattr(prov, "page_no", None)) is not None:
                return int(page_no)
    return None


def get_content_type(chunk) -> str:
    for item in getattr(chunk.meta, "doc_items", []):
        label = getattr(item, "label", None)
        label_name = getattr(label, "name", str(label))
        if label_name and label_name.upper() == "TABLE":
            return "table"
    return "text"


def generate_chunk_id(source: str, version: str, page: int | None, index: int, content: str) -> str:
    unique_text = "|".join([source, version, str(page), str(index), content])
    return hashlib.sha256(unique_text.encode("utf-8")).hexdigest()


def extract_effective_date(content: str) -> str:
    patterns = [
        r"effective\s+(?:date|from)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        r"effective\s+(?:date|from)\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if not match:
            continue
        value = match.group(1)
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                continue
    return ""


def chunk_docling_document(dl_doc, source: str, latest_year: str) -> list[Document]:
    if dl_doc is None:
        raise ValueError("DoclingDocument cannot be None.")

    chunker = create_hybrid_chunker()
    version = extract_year(source)
    status = get_document_status(version, latest_year)
    chunks = list(chunker.chunk(dl_doc=dl_doc))
    logger.info("Docling generated %d chunks for %s", len(chunks), source)

    documents = []
    for index, chunk in enumerate(chunks):
        content = chunker.contextualize(chunk).strip()
        if not content:
            continue

        page = extract_page_number(chunk)
        headings = getattr(chunk.meta, "headings", None) or []
        metadata = {
            "source": source,
            "version": version,
            "document_year": version,
            "page": int(page) if page is not None else -1,
            "document_type": "bank_document",
            "file_type": "pdf",
            "status": status,
            "content_type": get_content_type(chunk),
            "image_path": "",
            "effective_date": extract_effective_date(content),
            "visual_type": "",
            "numeric_facts": "[]",
            "headings": json.dumps(headings, ensure_ascii=False),
            "chunk_index": index,
            "chunking_method": "docling_hybrid",
        }
        metadata["chunk_id"] = generate_chunk_id(source, version, page, index, content)
        documents.append(Document(page_content=content, metadata=metadata))

    logger.info("Created %d LangChain documents from %s", len(documents), source)
    return documents


def print_chunk_statistics(documents: list[Document]) -> None:
    counts = {"text": 0, "table": 0}
    for doc in documents:
        c_type = doc.metadata.get("content_type", "text")
        counts[c_type] = counts.get(c_type, 0) + 1

    logger.info("DOCLING CHUNKING SUMMARY | Total chunks: %d | %s", len(documents), counts)
