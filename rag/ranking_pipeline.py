"""Retrieve -> version labelling -> rerank -> version rules -> confidence, in ONE place.

This is the hand-over point to the answer stage (Member 4):

    pipeline = build_ranking_pipeline(retriever)          # retriever = HybridRetriever(...)
    ctx = pipeline.ask("What is the Business Plus monthly fee?")
    ctx.docs                 # chunks that may be used in the prompt (best first, with effective_date)
    ctx.citations            # one dict per chunk: source, version, effective_date, page, ...
    ctx.flags                # see FLAG_* below and the FLAG_* in rag/versioning.py
    ctx.superseded_support   # old-version chunks, ONLY when flag superseded_only is set (label them "superseded")

Design rules
  * Never raises for retrieval/ranking problems: it returns flags instead (F9, fail safely and visibly).
  * Never sends a superseded chunk as current: old chunks are only exposed through ``superseded_support``.
  * No chunk text is logged (F6): only counts, flags and timings.
  * ``reranker=None`` gives the no-reranker baseline for the single-variable experiment.
  * The retriever only needs ``search(query, k=..., filters=...) -> list[Document]``
    (HybridRetriever.search matches). Do NOT filter on status/version at retrieval; versioning decides.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from langchain_core.documents import Document

from rag.reranking import KEY_RERANK_RANK, KEY_RERANK_SCORER, Reranker, build_reranker
from rag.versioning import (
    FLAG_RERANK_DEGRADED,
    FLAG_SUPERSEDED_ONLY,
    FLAG_VERSION_UNRESOLVED,
    KEY_CROSS_ENCODER_SCORE,
    RejectionReason,
    VersionCatalog,
    VersionService,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FLAG_NOT_FOUND = "not_found"                # nothing usable: refuse, never guess
FLAG_LOW_CONFIDENCE = "low_confidence"      # best rerank score below the configured floor
FLAG_RETRIEVAL_ERROR = "retrieval_error"    # the retriever failed
FLAG_RANKING_ERROR = "ranking_error"        # versioning/reranking failed
FLAG_EMPTY_QUESTION = "empty_question"
FLAG_SECURITY_BLOCKED = "security_blocked"  # some chunks were withheld for injection content


class Retriever(Protocol):
    def search(self, query: str, k: Optional[int] = None, **kwargs: Any) -> list[Document]: ...


@dataclass(frozen=True)
class RankingConfig:
    candidates: int = 15                    # chunks fetched from the retriever (brief: 10-20)
    final_k: int = 5                        # chunks handed to the LLM (brief: 3-5)
    min_relevance: Optional[float] = None   # raw cross-encoder floor; tune on golden questions, None = off

    def __post_init__(self) -> None:
        if self.candidates < 1 or self.final_k < 1:
            raise ValueError("candidates and final_k must be >= 1")
        if self.final_k > self.candidates:
            raise ValueError("final_k cannot be larger than candidates")


def _dedupe(docs: list[Document]) -> list[Document]:
    """Drop repeated chunks (keeps the first, i.e. best-ranked). Needed because the hybrid retriever can
    return the dense and BM25 copy of one chunk under two different keys when chunk_id is missing."""
    seen: set[tuple[Any, Any, str]] = set()
    unique: list[Document] = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("start_index"),
               hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest())
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _citation(doc: Document) -> dict[str, Any]:
    m = doc.metadata
    return {
        "source": m.get("source"),
        "version": m.get("version"),
        "effective_date": m.get("effective_date"),   # always shown with a fee (domain rule)
        "version_status": m.get("version_status"),
        "version_role": m.get("version_role"),       # current | historical (answerable chunks only)
        "page": m.get("page"),                       # None until ingestion records pages
        "section": m.get("section"),                 # None until ingestion records headings
        "doc_family": m.get("doc_family"),
        "content_type": m.get("content_type"),
        "chunk_index": m.get("chunk_index"),
        "start_index": m.get("start_index"),
        "rerank_rank": m.get(KEY_RERANK_RANK),
    }


@dataclass
class RankedContext:
    question: str
    docs: list[Document] = field(default_factory=list)
    superseded_support: list[Document] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    version_info: dict[str, Any] = field(default_factory=dict)
    blocked_count: int = 0
    scorer: Optional[str] = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return bool(self.docs)

    @property
    def citations(self) -> list[dict[str, Any]]:
        return [_citation(d) for d in self.docs]

    @property
    def superseded_citations(self) -> list[dict[str, Any]]:
        return [_citation(d) for d in self.superseded_support]

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        def pack(docs: list[Document]) -> list[dict[str, Any]]:
            return [{**_citation(d), **({"text": d.page_content} if include_text else {})} for d in docs]

        return {
            "question": self.question, "answerable": self.answerable, "flags": list(self.flags),
            "chunks": pack(self.docs), "superseded_support": pack(self.superseded_support),
            "version_info": self.version_info, "blocked_count": self.blocked_count,
            "scorer": self.scorer, "timings_ms": dict(self.timings_ms),
        }


class RankingPipeline:
    """Create once per process (the version catalog and the reranker model are reused)."""

    def __init__(self, retriever: Retriever, service: VersionService, reranker: Optional[Reranker] = None,
                 config: Optional[RankingConfig] = None) -> None:
        self._retriever = retriever
        self._service = service
        self._reranker = reranker
        self._cfg = config or RankingConfig()

    def ask(self, question: str, *, filters: Any = None, requested_as_of: Optional[date] = None) -> RankedContext:
        started = time.perf_counter()
        ctx = RankedContext(question=question or "")
        query = (question or "").strip()
        if not query:
            ctx.flags = [FLAG_EMPTY_QUESTION, FLAG_NOT_FOUND]
            return ctx

        # 1) retrieve candidates (more than we need; ranking decides)
        try:
            kwargs: dict[str, Any] = {"k": self._cfg.candidates}
            if filters is not None:
                kwargs["filters"] = filters
            candidates = self._retriever.search(query, **kwargs)
        except Exception as exc:  # noqa: BLE001 - must fail visibly, never crash the API
            logger.error("Retrieval failed: %s", type(exc).__name__)
            ctx.flags = [FLAG_RETRIEVAL_ERROR]
            return ctx
        ctx.timings_ms["retrieval"] = round((time.perf_counter() - started) * 1000, 1)
        candidates = _dedupe(list(candidates or []))
        if not candidates:
            ctx.flags = [FLAG_NOT_FOUND]
            return ctx

        # 2) label versions -> rerank -> version rules
        ranking_started = time.perf_counter()
        try:
            outcome = self._service.run(
                query, candidates, reranker=self._reranker, requested_as_of=requested_as_of,
                min_relevance=self._cfg.min_relevance, max_answerable=self._cfg.final_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Ranking failed: %s", type(exc).__name__)
            ctx.flags = [FLAG_RANKING_ERROR]
            return ctx
        ctx.timings_ms["ranking"] = round((time.perf_counter() - ranking_started) * 1000, 1)

        flags = list(outcome.flags)
        docs = list(outcome.answerable)
        ctx.blocked_count = len(outcome.security_blocked)
        if ctx.blocked_count:
            flags.append(FLAG_SECURITY_BLOCKED)

        # 3) confidence: weak best match -> refuse instead of guessing
        #    (skipped when the reranker degraded: the fallback scorer has a different score scale)
        if docs and self._cfg.min_relevance is not None and FLAG_RERANK_DEGRADED not in flags:
            scores = [float(d.metadata[KEY_CROSS_ENCODER_SCORE]) for d in docs
                      if d.metadata.get(KEY_CROSS_ENCODER_SCORE) is not None]
            if scores and max(scores) < self._cfg.min_relevance:
                docs = []
                flags.append(FLAG_LOW_CONFIDENCE)

        # 4) old versions are exposed ONLY when they are the sole support, clearly separated
        if FLAG_SUPERSEDED_ONLY in flags:
            ctx.superseded_support = [r.doc for r in outcome.rejected
                                      if r.reason is RejectionReason.SUPERSEDED][: self._cfg.final_k]

        if not docs and not ctx.superseded_support and FLAG_VERSION_UNRESOLVED not in flags:
            flags.append(FLAG_NOT_FOUND)

        ctx.docs, ctx.flags, ctx.version_info = docs, list(dict.fromkeys(flags)), outcome.version_info
        ctx.scorer = docs[0].metadata.get(KEY_RERANK_SCORER) if docs else None
        ctx.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 1)
        logger.info("Ranking: candidates=%d answerable=%d superseded_support=%d blocked=%d flags=%s total_ms=%.1f",
                    len(candidates), len(ctx.docs), len(ctx.superseded_support), ctx.blocked_count,
                    ctx.flags, ctx.timings_ms["total"])
        return ctx

    def as_runnable(self):
        """LCEL step: accepts a question string or {"question", "filters", "requested_as_of"}."""
        from langchain_core.runnables import RunnableLambda  # lazy: only needed inside a chain

        def _run(value: Any) -> RankedContext:
            if isinstance(value, str):
                return self.ask(value)
            return self.ask(value.get("question", ""), filters=value.get("filters"),
                            requested_as_of=value.get("requested_as_of"))

        return RunnableLambda(_run)


def build_ranking_pipeline(retriever: Retriever, *, processed_dir: "str | Path" = DEFAULT_PROCESSED_DIR,
                           config: Optional[RankingConfig] = None, use_reranker: bool = True,
                           use_cross_encoder: bool = True,
                           today_provider: Callable[[], date] = date.today) -> RankingPipeline:
    """Wire everything. ``use_reranker=False`` is the baseline for the single-variable experiment."""
    catalog = VersionCatalog.from_markdown_dir(processed_dir)
    service = VersionService(catalog, today_provider=today_provider)
    reranker = build_reranker(use_cross_encoder=use_cross_encoder) if use_reranker else None
    return RankingPipeline(retriever, service, reranker, config)
