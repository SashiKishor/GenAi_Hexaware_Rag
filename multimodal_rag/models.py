"""
Metadata and Document Factory for Multimodal RAG Ingestion.

Standardizes ChromaDB-safe metadata (versioning, status, effective dates, image paths,
and numeric facts), generates deterministic SHA-256 chunk IDs, and constructs
validated LangChain Document objects for text, tables, charts, and diagrams.

"""


import hashlib
import json
import re
from datetime import datetime
from typing import Any

from langchain_core.documents import Document


def extract_document_year(source: str) -> str:
    """Extract a four-digit year from the source filename."""
    match = re.search(r"20\d{2}", source)
    return match.group(0) if match else "unknown"


def determine_document_status(document_year: str, latest_year: str) -> str:
    """Determine version status using the available corpus."""
    if document_year == "unknown" or latest_year == "unknown":
        return "unknown"
    if document_year == latest_year:
        return "current"
    if int(document_year) < int(latest_year):
        return "superseded"
    return "unknown"


def extract_effective_date(text: str) -> str | None:
    """Extract an explicitly stated effective date in ISO format."""
    patterns = [
        r"effective\s+(?:date|from)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        r"effective\s+(?:date|from)\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip()
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def create_chunk_id(source: str, version: str, page: int | str, content_type: str, content: str) -> str:
    """Generate a deterministic SHA-256 ID for a document chunk."""
    unique_content = "|".join([source, str(version), str(page), content_type, content])
    return hashlib.sha256(unique_content.encode("utf-8")).hexdigest()


def serialize_numeric_facts(numeric_facts: list[dict[str, Any]]) -> str:
    """Serialize numeric facts into a Chroma-compatible JSON string."""
    return json.dumps(numeric_facts or [], ensure_ascii=False)


def build_metadata(
    source: str,
    page: int,
    content_type: str,
    content: str,
    latest_year: str,
    image_path: str | None = None,
    effective_date: str | None = None,
    numeric_facts: list[dict[str, Any]] | None = None,
    visual_type: str | None = None,
) -> dict:
    year = extract_document_year(source)
    metadata = {
        "source": source,
        "version": year,
        "document_year": year,
        "page": int(page),
        "document_type": "bank_document",
        "file_type": "pdf",
        "status": determine_document_status(year, latest_year),
        "content_type": content_type,
        "effective_date": effective_date or "",
        "image_path": image_path or "",
        "visual_type": visual_type or "",
        "numeric_facts": serialize_numeric_facts(numeric_facts or []),
    }
    metadata["chunk_id"] = create_chunk_id(source, year, page, content_type, content)
    return metadata


def create_document(content: str, metadata: dict) -> Document:
    """Create a LangChain Document with validated content."""
    if not content or not content.strip():
        raise ValueError("Document content cannot be empty.")
    if not metadata.get("chunk_id"):
        raise ValueError("Document metadata requires chunk_id.")
    return Document(page_content=content.strip(), metadata=metadata)
