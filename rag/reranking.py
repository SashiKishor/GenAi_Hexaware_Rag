"""Reranking stage (sits between version labelling and version rules).

    VersionService.run:  label chunks -> resolve versions -> **Reranker.rerank** -> version rules

Contract with ``rag/versioning.py`` (``VersionService.run`` calls exactly this):

    result = reranker.rerank(question, docs, temporal_intent="current" | "historical")
    result.ranked_docs    # ALL non-blocked chunks, best first, each carrying ``cross_encoder_score``
    result.blocked_docs   # chunks withheld for prompt-injection content (never reach the LLM)

Design rules
  * Relevance NEVER overrides version eligibility. The reranker only re-orders; it does not drop
    chunks for version reasons (``apply_version_rules`` does that afterwards on the full set).
  * The version nudge is soft and only applies to CURRENT questions (superseded/unresolved chunks
    are pushed down a little). Historical questions get no nudge, so old terms can rank on merit.
  * Injection-looking chunks are removed here (F7) and reported separately, never silently.
  * Fails safe (F9): if the cross-encoder breaks, fall back to a lexical scorer and mark the
    result ``degraded``; if that breaks too, keep the input order. It never raises on scoring.
  * No chunk text is logged (PII/F6): only counts, scorer names and pattern names.

Scorers (swap freely, they only need ``name``, ``nudge_scale`` and ``score``):
  * ``CrossEncoderScorer``  - sentence-transformers model, lazy-loaded (``pip install sentence-transformers``)
  * ``LexicalScorer``       - dependency-free BM25-style overlap; used for tests/offline/fallback
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Pattern, Protocol, Sequence

from langchain_core.documents import Document

from rag.versioning import (
    KEY_CROSS_ENCODER_SCORE,
    KEY_VERSION_STATUS,
    TemporalIntent,
    VersionStatus,
)

logger = logging.getLogger(__name__)

KEY_RERANK_SCORE = "rerank_score"        # raw score + soft version nudge (used for ordering)
KEY_RERANK_RANK = "rerank_rank"          # 1 = best
KEY_RERANK_SCORER = "rerank_scorer"      # which scorer produced cross_encoder_score (scales differ!)
KEY_SECURITY_FLAG = "security_flag"      # set on blocked chunks: name of the matched pattern

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# =============================================================================================
# Scorers
# =============================================================================================
class Scorer(Protocol):
    name: str
    nudge_scale: float  # multiplies the version nudge so it is comparable to this scorer's score range

    def score(self, question: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderScorer:
    """sentence-transformers CrossEncoder (raw logits, roughly -11..+11). Model loads on first use."""

    nudge_scale = 1.0

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL, batch_size: int = 16) -> None:
        self.name = f"cross-encoder:{model_name}"
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder  # lazy: heavy import

                    self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        model = self._load()
        scores = model.predict([(question, t) for t in texts], batch_size=self._batch_size)
        return [float(s) for s in scores]


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its of on or "
    "that the their this to was what when which who will with you your me my we our should".split()
)


def _tokens(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        out.append(tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok)  # crude plural folding
    return out


class LexicalScorer:
    """Dependency-free BM25-style scorer; result is in 0..1. Idf is computed over the candidate set."""

    name = "lexical"
    nudge_scale = 0.1  # score range is ~0..1, so a 1.0 nudge would swamp relevance
    _K = 1.2

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        q_terms = list(dict.fromkeys(_tokens(question)))
        if not q_terms or not texts:
            return [0.0] * len(texts)
        doc_tokens = [_tokens(t) for t in texts]
        n = len(doc_tokens)
        idf = {}
        for term in q_terms:
            df = sum(1 for toks in doc_tokens if term in toks)
            idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        total = sum(idf.values()) or 1.0
        out = []
        for toks in doc_tokens:
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            raw = sum(idf[t] * (counts[t] * (self._K + 1)) / (counts[t] + self._K) for t in q_terms if t in counts)
            out.append(raw / (total * (self._K + 1)))
        return out


# =============================================================================================
# Injection screening (F7) - cheap deterministic first line; guardrail layers come later
# =============================================================================================
_INJECTION_PATTERNS: dict[str, Pattern[str]] = {
    name: re.compile(rx, re.IGNORECASE)
    for name, rx in {
        "ignore_instructions": r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:instructions?|rules?|polic(?:y|ies)|prompts?|guidelines?)\b",
        "reveal_secrets": r"\b(?:reveal|disclose|print|leak|expose|output)\b[^.\n]{0,40}\b(?:system prompt|account numbers?|passwords?|api keys?|national ids?|customer data|pii)\b",
        "role_hijack": r"\b(?:you are now|act as (?:an?|the)|pretend (?:to be|you are)|new instructions?)\b",
        "role_marker": r"(?:^|\n)\s*(?:system|assistant)\s*:\s*\S",
        "chat_template_token": r"<\|(?:im_start|im_end|system|assistant)\|>|\[/?INST\]|<<SYS>>",
        "tool_exfiltration": r"\b(?:send|post|forward|email|upload)\b[^.\n]{0,40}\b(?:to|at)\b[^.\n]{0,40}(?:https?://|@\w)",
    }.items()
}

_NEGATION = re.compile(r"\b(?:not|never|cannot|can't|won't|shall not|must not|without|refuse to|avoid)\b|n't\b", re.IGNORECASE)


def _negated(text: str, start: int) -> bool:
    """True when the match is a prohibition ('must not disclose ...'), i.e. policy text, not an attack."""
    return bool(_NEGATION.search(text[max(0, start - 30):start]))


def detect_injection(text: str) -> Optional[str]:
    """Name of the first matching injection pattern, or None. Returns the NAME only (never text).

    Imperatives are flagged; prohibitions that merely restate policy ("must not disclose the account
    number") are not, otherwise the bank's own rules would be blocked as attacks.
    """
    for name, pattern in _INJECTION_PATTERNS.items():
        for match in pattern.finditer(text):
            if not _negated(text, match.start()):
                return name
    return None


# =============================================================================================
# Reranker
# =============================================================================================
@dataclass(frozen=True)
class RerankerConfig:
    top_k: Optional[int] = None          # None = return everything (REQUIRED for VersionService: rules need the full set)
    max_chars: int = 2000                # text sent to the scorer (cross-encoders truncate anyway)
    current_bonus: float = 0.5           # nudge for status=current          (current-intent only)
    superseded_penalty: float = 0.5      # nudge for status=superseded/future (current-intent only)
    unresolved_penalty: float = 0.25     # nudge for status=unresolved        (current-intent only)
    block_injection: bool = True

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 or None")
        if min(self.current_bonus, self.superseded_penalty, self.unresolved_penalty) < 0:
            raise ValueError("bonus/penalties are magnitudes and must be >= 0")
        if self.max_chars < 1:
            raise ValueError("max_chars must be >= 1")


@dataclass
class RerankResult:
    ranked_docs: list[Document] = field(default_factory=list)
    blocked_docs: list[Document] = field(default_factory=list)
    scorer: str = "none"
    degraded: bool = False
    degraded_reason: Optional[str] = None
    temporal_intent: str = TemporalIntent.CURRENT.value
    elapsed_ms: float = 0.0              # time spent in rerank(), for the latency evidence


class Reranker:
    """Scores chunks against the question, applies a soft version nudge, sorts, never filters by version."""

    def __init__(self, scorer: Optional[Scorer] = None, *, fallback: Optional[Scorer] = None,
                 config: Optional[RerankerConfig] = None) -> None:
        self._scorer: Scorer = scorer or LexicalScorer()
        self._fallback: Optional[Scorer] = fallback if fallback is not None else (
            LexicalScorer() if not isinstance(self._scorer, LexicalScorer) else None)
        self._cfg = config or RerankerConfig()

    # -- scoring with graceful degradation -------------------------------------------------
    def _score(self, question: str, texts: list[str]) -> tuple[list[float], Scorer, Optional[str]]:
        reason: Optional[str] = None
        for scorer in (self._scorer, self._fallback):
            if scorer is None:
                continue
            try:
                scores = scorer.score(question, texts)
                if len(scores) != len(texts) or any(not math.isfinite(s) for s in scores):
                    raise ValueError("scorer returned wrong length or non-finite values")
                return scores, scorer, reason
            except Exception as exc:  # noqa: BLE001 - must never break the pipeline
                reason = f"{scorer.name} failed: {type(exc).__name__}"
                logger.warning("Reranker scorer failed (%s); trying fallback", reason)
        return [], self._scorer, reason or "no scorer available"

    def _nudge(self, status: Optional[str], intent: str) -> float:
        if intent != TemporalIntent.CURRENT.value:
            return 0.0  # historical question: old terms must compete on relevance alone
        if status == VersionStatus.CURRENT.value:
            return self._cfg.current_bonus
        if status in (VersionStatus.SUPERSEDED.value, VersionStatus.FUTURE.value):
            return -self._cfg.superseded_penalty
        if status == VersionStatus.UNRESOLVED.value:
            return -self._cfg.unresolved_penalty
        return 0.0

    # -- public API -------------------------------------------------------------------------
    def rerank(self, question: str, docs: Sequence[Document], *, temporal_intent: str = "current") -> RerankResult:
        started = time.perf_counter()
        result = self._rerank(question, docs, temporal_intent=temporal_intent)
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info("Rerank: scorer=%s ranked=%d blocked=%d degraded=%s elapsed_ms=%.1f", result.scorer,
                    len(result.ranked_docs), len(result.blocked_docs), result.degraded, result.elapsed_ms)
        return result

    def _rerank(self, question: str, docs: Sequence[Document], *, temporal_intent: str = "current") -> RerankResult:
        intent = TemporalIntent(temporal_intent).value  # raises on typos instead of silently mis-ranking
        result = RerankResult(temporal_intent=intent)

        safe: list[Document] = []
        for doc in docs:
            flag = detect_injection(doc.page_content) if self._cfg.block_injection else None
            if flag:
                result.blocked_docs.append(doc.model_copy(update={"metadata": {**doc.metadata, KEY_SECURITY_FLAG: flag}}))
            else:
                safe.append(doc)
        if result.blocked_docs:
            logger.warning("Reranker blocked %d chunk(s) for injection patterns: %s", len(result.blocked_docs),
                           sorted({d.metadata[KEY_SECURITY_FLAG] for d in result.blocked_docs}))
        if not safe:
            return result

        texts = [d.page_content[: self._cfg.max_chars] for d in safe]
        scores, scorer, reason = self._score(question, texts)
        result.scorer = scorer.name

        if not scores:  # every scorer failed: keep input order, scores unknown (rules treat None as supporting)
            result.degraded, result.degraded_reason = True, reason
            result.ranked_docs = safe[: self._cfg.top_k]
            return result
        if reason:  # primary failed, fallback worked
            result.degraded, result.degraded_reason = True, reason

        scored = []
        for idx, (doc, raw) in enumerate(zip(safe, scores)):
            adjusted = raw + self._nudge(doc.metadata.get(KEY_VERSION_STATUS), intent) * scorer.nudge_scale
            scored.append((adjusted, idx, raw, doc))
        scored.sort(key=lambda t: (-t[0], t[1]))  # stable: ties keep retrieval order

        for rank, (adjusted, _idx, raw, doc) in enumerate(scored[: self._cfg.top_k], start=1):
            result.ranked_docs.append(doc.model_copy(update={"metadata": {
                **doc.metadata, KEY_CROSS_ENCODER_SCORE: raw, KEY_RERANK_SCORE: adjusted,
                KEY_RERANK_RANK: rank, KEY_RERANK_SCORER: scorer.name}}))
        return result


def build_reranker(*, use_cross_encoder: bool = True, model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
                   config: Optional[RerankerConfig] = None) -> Reranker:
    """Cross-encoder if installed, otherwise lexical (logged). Model weights load on first rerank."""
    if use_cross_encoder:
        try:
            import sentence_transformers  # noqa: F401

            return Reranker(CrossEncoderScorer(model_name), config=config)
        except ImportError:
            logger.warning("sentence-transformers not installed; using the lexical reranker")
    return Reranker(LexicalScorer(), config=config)


# =============================================================================================
# Demo:  python -m rag.reranking   (runs the whole label -> resolve -> rerank -> rules flow)
# =============================================================================================
def _demo() -> None:  # pragma: no cover - convenience only
    from datetime import date
    from pathlib import Path

    from langchain_core.documents import Document as Doc

    from ingestion.chunking import split_documents
    from rag.versioning import VersionCatalog, VersionService

    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    catalog = VersionCatalog.from_markdown_dir(data_dir)
    pages = [Doc(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
             for p in sorted(data_dir.glob("*.md"))]
    chunks = split_documents(pages, chunk_size=1000, chunk_overlap=150)

    service = VersionService(catalog, today_provider=lambda: date(2026, 11, 19))
    reranker = build_reranker(use_cross_encoder=False)
    for question in ("What is the Business Plus monthly fee?",
                     "Does the 2025 overdraft policy still apply?"):
        out = service.run(question, chunks, reranker=reranker)
        print(f"\nQ: {question}\n  intent={out.temporal.intent.value}  flags={out.flags}  "
              f"answerable={len(out.answerable)}  rejected={len(out.rejected)}  blocked={len(out.security_blocked)}")
        for d in out.answerable[:3]:
            m = d.metadata
            print(f"  #{m['rerank_rank']} {m['source']} v{m['version']} ({m['version_role']}) "
                  f"score={m['cross_encoder_score']:.3f}")


if __name__ == "__main__":  # pragma: no cover
    _demo()