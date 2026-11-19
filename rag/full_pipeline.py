"""
End-to-End Orchestrator for Northbridge Bank RAG.

Runs text retrieval/ranking and multimodal visual/numeric retrieval in parallel,
merges deduplicated visual chunks into RankedContext, and generates the final
compliance-grounded answer (or token stream) for FastAPI.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, AsyncIterator

from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel

from multimodal_rag import MultimodalVisualRetriever
from rag.chain import generate_answer, stream_answer
from rag.ranking_pipeline import RankedContext, RankingPipeline

logger = logging.getLogger(__name__)


def _merge_contexts(text_ctx: RankedContext, visual_docs: list[Document]) -> RankedContext:
    """Safely appends unique visual/table chunks to RankedContext without duplicating text chunks."""
    if not visual_docs:
        return text_ctx

    seen_hashes = {
        hashlib.sha256(d.page_content.encode("utf-8")).hexdigest()
        for d in (text_ctx.docs + text_ctx.superseded_support)
    }
    seen_ids = {
        d.metadata.get("chunk_id")
        for d in (text_ctx.docs + text_ctx.superseded_support)
        if d.metadata.get("chunk_id")
    }

    for doc in visual_docs:
        cid = doc.metadata.get("chunk_id")
        chash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        if (cid and cid in seen_ids) or chash in seen_hashes:
            continue

        # Respect version safety: only add current/matching visuals to active docs
        status = str(doc.metadata.get("status", "")).lower()
        if status == "superseded" and text_ctx.docs:
            continue

        if status == "superseded" and not text_ctx.docs:
            text_ctx.superseded_support.append(doc)
        else:
            text_ctx.docs.append(doc)
            # Clear 'not_found' if visual/numeric search found valid current evidence
            if "not_found" in text_ctx.flags:
                text_ctx.flags.remove("not_found")

        if cid:
            seen_ids.add(cid)
        seen_hashes.add(chash)

    return text_ctx


class NorthbridgeRAGPipeline:
    """Single entry point for FastAPI: pass `question` in, get final answer + citations out."""

    def __init__(self, ranking_pipeline: RankingPipeline, llm: Any, visual_top_k: int = 3) -> None:
        self.ranking_pipeline = ranking_pipeline
        self.visual_rag = MultimodalVisualRetriever(top_k=visual_top_k, max_images=3)
        self.llm = llm

        # Run text ranking and multimodal visual/numeric retrieval in parallel
        self.parallel_retriever = RunnableParallel(
            text_ctx=self.ranking_pipeline.as_runnable(),
            visual_docs=self.visual_rag.retriever.retrieve,  # returns list[Document]
        )

    def _retrieve_combined(self, question: str) -> RankedContext:
        out = self.parallel_retriever.invoke(question)
        return _merge_contexts(out["text_ctx"], out["visual_docs"])

    async def ask(self, question: str) -> dict[str, Any]:
        """Non-streaming endpoint handler (POST /ask)."""
        ctx = self._retrieve_combined(question)
        result = await generate_answer(question=question, ctx=ctx, llm=self.llm)
        result["images"] = [
            {
                "source": c["document_id"],
                "page": c["page_or_section"],
                "content_type": c["content_type"],
                "image_path": c["image_path"],
            }
            for c in result["citations"]
            if c.get("image_path")
        ]
        return result

    async def stream(self, question: str) -> AsyncIterator[dict[str, Any]]:
        """Streaming endpoint handler (POST /ask/stream)."""
        ctx = self._retrieve_combined(question)
        async for event in stream_answer(question=question, ctx=ctx, llm=self.llm):
            yield event
