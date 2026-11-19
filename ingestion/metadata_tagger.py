"""Adds doc_family, content_type and effective_date to chunks so the
retriever can filter on them. Chunks must be passed in document order."""

import re

from langchain_core.documents import Document

# Section-title pattern -> doc_family. First match wins.
FAMILY_RULES = [
    (r"document control", "document_control"),
    (r"kyc|aml", "kyc_aml"),
    (r"fee schedule", "fee_schedule"),
    (r"branch process", "branch_process"),
    (r"correspondence|letter", "customer_letter"),
    (r"validation|glossary", "reference"),
    (r"product terms", "product_terms"),
]

# Taken from each pack's document-control page.
EFFECTIVE_DATES = {
    "2025": "2025-01-01",
    "2026": "2026-01-01",
}

HEADING_RE = re.compile(r"^\s*(?:#+\s*)?\d{1,2}\.\s+(.+)$", re.MULTILINE)
HEADING_AT_START = 100  # a heading this close to the top labels the whole chunk
TABLE_MIN_LINES = 8
TABLE_MAX_AVG_LINE = 28  # pypdf flattens tables into short one-cell lines


def _family_from_title(title: str) -> str | None:
    for pattern, family in FAMILY_RULES:
        if re.search(pattern, title, re.IGNORECASE):
            return family
    return None


def _content_type(text: str) -> str:
    if "<!-- image -->" in text:
        return "image"

    lines = [line for line in text.splitlines() if line.strip()]

    if sum(line.lstrip().startswith("|") for line in lines) >= 2:
        return "table"

    if len(lines) >= TABLE_MIN_LINES:
        if sum(len(line) for line in lines) / len(lines) <= TABLE_MAX_AVG_LINE:
            return "table"

    return "text"


def tag_chunks(chunks: list[Document]) -> list[Document]:
    """Tag chunks in place and return them."""
    current_document = None
    section_family = "general"

    for chunk in chunks:
        document_id = (chunk.metadata.get("source"), chunk.metadata.get("version"))
        if document_id != current_document:
            current_document = document_id
            section_family = "general"

        chunk_family = section_family
        label_set = False

        for match in HEADING_RE.finditer(chunk.page_content):
            family = _family_from_title(match.group(1))
            if family is None:  # numbered table rows such as "1. Identity"
                continue
            if match.start() <= HEADING_AT_START and not label_set:
                chunk_family = family
                label_set = True
            section_family = family

        chunk.metadata["doc_family"] = chunk_family
        chunk.metadata["content_type"] = _content_type(chunk.page_content)
        chunk.metadata["effective_date"] = EFFECTIVE_DATES.get(
            str(chunk.metadata.get("version")), "unknown"
        )

    return chunks