"""Hybrid retriever: dense (ChromaDB) search plus BM25 keyword search,
merged with Reciprocal Rank Fusion (RRF).

    retriever = HybridRetriever(ChromaDB().get_vector_store())
    results = retriever.retrieve(
        "Business Plus monthly fee",
        filters=MetadataFilter(status="current"),
    )
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

FilterValue = str | Sequence[str]
FILTER_FIELDS = ("status", "doc_family", "content_type", "version")


@dataclass(frozen=True)
class RetrieverConfig:
    top_k: int = 5
    dense_candidates: int = 20
    bm25_candidates: int = 20
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0

    def __post_init__(self):
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.dense_candidates <= 0 or self.bm25_candidates <= 0:
            raise ValueError("Candidate counts must be greater than zero.")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")
        if self.dense_weight < 0 or self.bm25_weight < 0:
            raise ValueError("Weights cannot be negative.")


def _as_list(value: FilterValue) -> list[str]:
    return [value] if isinstance(value, str) else [str(v) for v in value]


@dataclass(frozen=True)
class MetadataFilter:
    """Restrict results by metadata. Each field takes one value or a list."""

    status: FilterValue | None = None
    doc_family: FilterValue | None = None
    content_type: FilterValue | None = None
    version: FilterValue | None = None

    def __post_init__(self):
        for name, values in self._conditions().items():
            if not values:
                raise ValueError(f"Filter '{name}' has no values.")

    def _conditions(self) -> dict[str, list[str]]:
        return {
            name: _as_list(getattr(self, name))
            for name in FILTER_FIELDS
            if getattr(self, name) is not None
        }

    def to_chroma(self) -> dict | None:
        clauses = [
            {name: values[0]} if len(values) == 1 else {name: {"$in": values}}
            for name, values in self._conditions().items()
        ]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def matches(self, metadata: dict) -> bool:
        return all(
            str(metadata.get(name)) in values
            for name, values in self._conditions().items()
        )


@dataclass(frozen=True)
class RetrievedChunk:
    document: Document
    rrf_score: float
    dense_rank: int | None = None
    dense_distance: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    rrf_k: int = 60,
) -> dict[str, float]:
    """Fuse ranked key lists (best first) into {key: score}."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
    return scores


def _chunk_key(document: Document) -> str:
    chunk_id = document.metadata.get("chunk_id")
    if chunk_id:
        return chunk_id
    digest = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
    return f"{document.metadata.get('source')}|{document.metadata.get('page')}|{digest}"


def _tokenize(text: str) -> list[str]:
    # Join digit groups so "2,00,000" and "50,000" become single tokens
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return re.findall(r"[a-z0-9]+", text)


class HybridRetriever:

    def __init__(self, vector_store, config: RetrieverConfig | None = None):
        self.vector_store = vector_store
        self.config = config or RetrieverConfig()
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[Document] = []

    def refresh_index(self) -> int:
        """Rebuild the BM25 index from the vector store.

        Called automatically on first use; call it again after re-ingesting.
        """
        data = self.vector_store.get(include=["documents", "metadatas"])

        self._bm25_docs = [
            Document(
                page_content=text,
                metadata={**(metadata or {}), "chunk_id": chroma_id},
            )
            for text, metadata, chroma_id in zip(
                data["documents"], data["metadatas"], data["ids"]
            )
        ]

        if not self._bm25_docs:
            self._bm25 = None
            logger.warning("Vector store is empty; BM25 index not built.")
            return 0

        self._bm25 = BM25Okapi(
            [_tokenize(doc.page_content) for doc in self._bm25_docs]
        )
        logger.info("BM25 index built with %d chunks", len(self._bm25_docs))
        return len(self._bm25_docs)

    def _dense_search(
        self, query: str, filters: MetadataFilter | None
    ) -> list[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(
            query,
            k=self.config.dense_candidates,
            filter=filters.to_chroma() if filters else None,
        )

    def _bm25_search(
        self, query: str, filters: MetadataFilter | None
    ) -> list[tuple[Document, float]]:
        if self._bm25 is None:
            self.refresh_index()
        tokens = _tokenize(query)
        if self._bm25 is None or not tokens:
            return []

        scored = [
            (doc, float(score))
            for doc, score in zip(self._bm25_docs, self._bm25.get_scores(tokens))
            if score > 0 and (filters is None or filters.matches(doc.metadata))
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.config.bm25_candidates]

    def retrieve(
        self,
        query: str,
        filters: MetadataFilter | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        top_k = self.config.top_k if top_k is None else top_k
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        dense = self._dense_search(query, filters)
        sparse = self._bm25_search(query, filters)

        # Only sizes are logged: the query may contain customer data.
        logger.info(
            "Retrieval: dense=%d bm25=%d query_length=%d",
            len(dense), len(sparse), len(query),
        )

        dense_hits = {
            _chunk_key(doc): (rank, doc, distance)
            for rank, (doc, distance) in enumerate(dense, start=1)
        }
        bm25_hits = {
            _chunk_key(doc): (rank, doc, score)
            for rank, (doc, score) in enumerate(sparse, start=1)
        }

        fused = reciprocal_rank_fusion(
            [list(dense_hits), list(bm25_hits)],
            weights=[self.config.dense_weight, self.config.bm25_weight],
            rrf_k=self.config.rrf_k,
        )

        chunks = []
        for key, rrf_score in fused.items():
            dense_rank, dense_doc, distance = dense_hits.get(key, (None, None, None))
            bm25_rank, bm25_doc, bm25_score = bm25_hits.get(key, (None, None, None))
            chunks.append(
                RetrievedChunk(
                    document=dense_doc or bm25_doc,
                    rrf_score=rrf_score,
                    dense_rank=dense_rank,
                    dense_distance=None if distance is None else float(distance),
                    bm25_rank=bm25_rank,
                    bm25_score=bm25_score,
                )
            )

        # Equal scores are ordered by dense rank so results are deterministic.
        chunks.sort(
            key=lambda c: (
                -c.rrf_score,
                c.dense_rank if c.dense_rank is not None else float("inf"),
                _chunk_key(c.document),
            )
        )
        return chunks[:top_k]

    def search(
        self,
        query: str,
        k: int | None = None,
        filters: MetadataFilter | None = None,
    ) -> list[Document]:
        """Documents only; same shape as vector_store.retriever.Retriever."""
        return [c.document for c in self.retrieve(query, filters, top_k=k)]

    def as_langchain_retriever(
        self, filters: MetadataFilter | None = None
    ) -> "LangChainHybridRetriever":
        return LangChainHybridRetriever(hybrid=self, filters=filters)


def only_superseded(results: list[RetrievedChunk]) -> bool:
    """True when every result comes from a superseded document."""
    return bool(results) and all(
        r.document.metadata.get("status") == "superseded" for r in results
    )


class LangChainHybridRetriever(BaseRetriever):
    """Adapter so the hybrid retriever can be used in LangChain chains."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid: HybridRetriever
    filters: MetadataFilter | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            Document(
                page_content=chunk.document.page_content,
                metadata={**chunk.document.metadata, "rrf_score": chunk.rrf_score},
            )
            for chunk in self.hybrid.retrieve(query, self.filters)
        ]