"""
Hybrid Multimodal Retriever for Northbridge Bank.

Combines dense vector search with exact and range-based numeric matching across text,
table, chart, and diagram chunks in ChromaDB, reranking results using lexical overlap,
numeric range containment, modality intent, and document version status (current vs. superseded).
"""


import json
import logging
import re
from typing import Optional

from langchain_core.documents import Document
from vector_store.chroma_db import ChromaDB

logger = logging.getLogger(__name__)

DEFAULT_K = 3
CONTENT_TYPES = {"text", "table", "chart", "diagram"}
STOP_WORDS = {
    "which", "what", "when", "where", "who", "is", "are", "was", "were",
    "the", "a", "an", "at", "of", "for", "to", "in", "on", "and", "or",
    "does", "do", "how", "me", "my", "this", "that", "with", "applies",
    "apply", "show", "tell",
}

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
NUMBER_TOKEN = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"


# 1. Numeric extraction & expansion
def extract_numbers(text: str) -> list[float]:
    if not text:
        return []
    numbers = []
    for match in NUMBER_PATTERN.findall(text):
        try:
            numbers.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return numbers


def expand_numeric_query(query: str) -> list[str]:
    expanded = [query]
    for number in extract_numbers(query):
        if number.is_integer():
            val = int(number)
            fmt = f"{val:,}"
            expanded.extend([
                str(val), fmt, f"₹{fmt}", f"balance {fmt}", f"average balance {fmt}",
                f"threshold {fmt}", f"minimum balance {fmt}", f"account tier {fmt}",
            ])
        else:
            expanded.extend([str(number), f"{number:,.2f}"])
    return list(dict.fromkeys(expanded))


# 2. Query understanding
def detect_requested_year(query: str) -> Optional[str]:
    match = re.search(r"\b(20\d{2})\b", query)
    return match.group(1) if match else None


def detect_comparison_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in ("compare", "comparison", "difference", "versus", "vs", "between", "both years"))


def detect_content_intent(query: str) -> set[str]:
    q = query.lower()
    intents = set()
    if any(w in q for w in ("tier", "balance", "threshold", "fee", "charge", "amount", "waiver", "schedule", "rate", "minimum")):
        intents.update(["table", "chart"])
    if any(w in q for w in ("diagram", "flowchart", "process", "workflow", "steps", "procedure", "sequence")):
        intents.add("diagram")
    if any(w in q for w in ("explain", "describe", "policy", "eligibility", "documents", "requirements", "kyc")):
        intents.add("text")
    return intents


def get_query_terms(query: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", query.lower())) - STOP_WORDS


# 3. Metadata & document helpers
def parse_numeric_facts(metadata: dict) -> list:
    facts = metadata.get("numeric_facts", "[]")
    if isinstance(facts, list):
        return facts
    try:
        parsed = json.loads(facts)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


def get_document_text(document: Document) -> str:
    facts = parse_numeric_facts(document.metadata)
    return f"{document.page_content} {' '.join(str(f) for f in facts)}"


# 4. Financial range matching
def to_number(value: str) -> float:
    return float(value.replace(",", ""))


def parse_range_matches(text: str) -> list[tuple[float, float]]:
    ranges = []
    pattern = rf"(?<![\w.])({NUMBER_TOKEN})\s*(?:-|–|—|to|through)\s*({NUMBER_TOKEN})(?![\w.])"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        try:
            low, high = to_number(match.group(1)), to_number(match.group(2))
            if low <= high:
                ranges.append((low, high))
        except ValueError:
            continue
    return ranges


def value_matches_range(value: float, text: str) -> bool:
    for low, high in parse_range_matches(text):
        if low <= value <= high:
            return True

    lower_patterns = [
        rf"({NUMBER_TOKEN})\s*(?:and above|or above|and over|or more|and higher|\+)",
        rf"(?:>=|at least|minimum of)\s*({NUMBER_TOKEN})",
    ]
    for pattern in lower_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if value >= to_number(match.group(1)):
                return True

    for match in re.finditer(rf"(?:up to|at most|below|less than)\s*({NUMBER_TOKEN})", text, re.IGNORECASE):
        if value <= to_number(match.group(1)):
            return True

    return False


def has_numeric_match(query_numbers: list[float], document_text: str) -> bool:
    doc_numbers = extract_numbers(document_text)
    return any(num in doc_numbers or value_matches_range(num, document_text) for num in query_numbers)


# 5. Unified Retriever
class UnifiedRetriever:
    def __init__(
        self,
        vector_store=None,
        chroma_db=None,
        top_k: int = DEFAULT_K,
        candidate_k: int = 10,
        latest_year: Optional[int] = None,
    ):
        if vector_store is None:
            chroma_db = chroma_db or ChromaDB()
            vector_store = chroma_db.get_vector_store()
        self.vector_store = vector_store
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.latest_year = latest_year
        logger.info("UnifiedRetriever initialized.")

    def semantic_search(self, query: str, k: int) -> list[Document]:
        results = []
        for expanded_query in expand_numeric_query(query)[:5]:
            try:
                for doc, dist in self.vector_store.similarity_search_with_score(expanded_query, k=k):
                    doc.metadata["_semantic_score"] = 1.0 / (1.0 + max(float(dist), 0.0))
                    results.append(doc)
            except Exception:
                logger.exception("Semantic search failed: %s", expanded_query)
        return results

    def numeric_search(self, query: str) -> list[Document]:
        query_numbers = extract_numbers(query)
        if not query_numbers:
            return []

        try:
            stored = self.vector_store.get(include=["documents", "metadatas"])
        except Exception:
            logger.exception("Numeric search failed.")
            return []

        documents = []
        texts = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []
        ids = stored.get("ids") or []

        for idx, content in enumerate(texts):
            if not content:
                continue
            metadata = dict(metadatas[idx]) if idx < len(metadatas) and metadatas[idx] else {}
            doc = Document(page_content=content, metadata=metadata)
            if has_numeric_match(query_numbers, get_document_text(doc)):
                doc.metadata["_numeric_match"] = True
                if idx < len(ids):
                    doc.metadata["_retrieval_id"] = ids[idx]
                documents.append(doc)
        return documents

    @staticmethod
    def deduplicate(documents: list[Document]) -> list[Document]:
        unique = {}
        for doc in documents:
            identity = doc.metadata.get("chunk_id") or (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.metadata.get("content_type"),
                doc.metadata.get("image_path"),
                doc.page_content,
            )
            if identity not in unique:
                unique[identity] = doc
            else:
                existing = unique[identity]
                existing.metadata["_semantic_score"] = max(
                    existing.metadata.get("_semantic_score", 0),
                    doc.metadata.get("_semantic_score", 0),
                )
                if doc.metadata.get("_numeric_match"):
                    existing.metadata["_numeric_match"] = True
        return list(unique.values())

    def resolve_latest_year(self, documents: list[Document]) -> Optional[int]:
        if self.latest_year is not None:
            return self.latest_year

        current_years, all_years = [], []
        for doc in documents:
            meta = doc.metadata
            try:
                year = int(meta.get("document_year", meta.get("version")))
            except (ValueError, TypeError):
                continue
            all_years.append(year)
            if meta.get("status") == "current":
                current_years.append(year)

        available = current_years or all_years
        return max(available) if available else None

    def rank_documents(
        self,
        documents: list[Document],
        query: str,
        latest_year: Optional[int] = None,
    ) -> list[Document]:
        query_terms = get_query_terms(query)
        query_numbers = extract_numbers(query)
        requested_year = detect_requested_year(query)
        comparison_query = detect_comparison_query(query)
        expected_types = detect_content_intent(query)
        latest_year = latest_year if latest_year is not None else self.resolve_latest_year(documents)

        scored_documents = []
        for doc in documents:
            meta = doc.metadata
            content = get_document_text(doc).lower()
            content_terms = set(re.findall(r"\b\w+\b", content))

            lexical_score = (len(query_terms & content_terms) / max(len(query_terms), 1)) * 10
            semantic_score = max(0.0, min(float(meta.get("_semantic_score", 0)), 1.0)) * 5
            numeric_score = 8.0 if (query_numbers and has_numeric_match(query_numbers, content)) else 0.0

            ctype = meta.get("content_type", "text")
            type_score = 3.0 if ctype in expected_types else 0.0

            try:
                year = int(meta.get("document_year", meta.get("version")))
            except (ValueError, TypeError):
                year = None

            version_score = 0.0
            if requested_year:
                version_score = 8.0 if year == int(requested_year) else 0.0
            elif comparison_query:
                version_score = 2.0
            elif latest_year and year == latest_year:
                version_score = 4.0

            status_score = 0.0
            if not (requested_year or comparison_query):
                status = meta.get("status")
                status_score = 2.0 if status == "current" else -1.0 if status == "superseded" else 0.0

                table_score = 2.0 if (query_numbers and ctype in {"table", "chart"}) else 0.0
                visual_boost = 2.5 if meta.get("image_path") else 0.0
                letter_penalty = -2.0 if ("dear " in content or "customer letter" in content) else 0.0

            total_score = (
                lexical_score
                + semantic_score
                + numeric_score
                + type_score
                + version_score
                + status_score
                + table_score
                + visual_boost
                + letter_penalty
            )


            meta["_rerank_score"] = round(total_score, 4)
            meta["_rerank_breakdown"] = {
                "lexical": round(lexical_score, 2),
                "semantic": round(semantic_score, 2),
                "numeric": numeric_score,
                "content_type": type_score,
                "version": version_score,
                "status": status_score,
                "table_chart": table_score,
            }
            scored_documents.append(doc)

        return sorted(scored_documents, key=lambda d: d.metadata.get("_rerank_score", 0), reverse=True)

    def retrieve(self, query: str, k: Optional[int] = None) -> list[Document]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        k = k or self.top_k
        logger.info("Retrieving for query: %s", query)

        semantic_docs = self.semantic_search(query, k=self.candidate_k)
        numeric_docs = self.numeric_search(query)
        unique_docs = self.deduplicate(semantic_docs + numeric_docs)
        latest_year = self.resolve_latest_year(unique_docs)
        final_docs = self.rank_documents(unique_docs, query, latest_year=latest_year)[:k]

        logger.info(
            "Retrieval complete | Semantic: %d | Numeric: %d | Unique: %d | Final: %d | Latest year: %s",
            len(semantic_docs),
            len(numeric_docs),
            len(unique_docs),
            len(final_docs),
            latest_year,
        )
        return final_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    retriever = UnifiedRetriever(top_k=6, candidate_k=20)

    test_queries = [
        "Which tier applies at an average balance of 45,000?",
        "Which tier applied in 2025 at an average balance of 45,000?",
        "Compare the 2025 and 2026 tier thresholds.",
        "What is the Business Plus monthly fee?",
        "What was the fee schedule in 2025?",
        "What documents are required for KYC?",
        "Explain the account opening process.",
    ]

    for query in test_queries:
        logger.info("\n%s\nQUERY: %s\n%s", "=" * 90, query, "=" * 90)
        for idx, doc in enumerate(retriever.retrieve(query), start=1):
            meta = doc.metadata
            logger.info(
                "\nRESULT %d | Type: %s | Source: %s | Version: %s | Page: %s | Image: %s | Score: %s | Numeric: %s\nBreakdown: %s\nContent: %s",
                idx,
                meta.get("content_type"),
                meta.get("source"),
                meta.get("version"),
                meta.get("page"),
                meta.get("image_path"),
                meta.get("_rerank_score"),
                meta.get("_numeric_match", False),
                meta.get("_rerank_breakdown"),
                doc.page_content[:500],
            )