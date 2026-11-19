"""
Visual & Numeric Retrieval Adapter for Northbridge Bank LCEL Chain.

Accepts a user query, retrieves relevant table, chart, and diagram chunks
(plus numeric range matches), and returns formatted visual text context,
multimodal image content blocks, and page/image citations for downstream merging.
"""

from langchain_core.runnables import RunnableLambda

from multimodal_rag.context_builder import ContextBuilder
from multimodal_rag.multimodal_retrieval import UnifiedRetriever


class MultimodalVisualRetriever:
    def __init__(self, top_k: int = 3, max_images: int = 3, include_table_images: bool = False):
        self.retriever = UnifiedRetriever(top_k=top_k, candidate_k=10)
        self.context_builder = ContextBuilder(
            max_images=max_images,
            include_table_images=include_table_images,
        )

    def get_visual_context(self, query: str) -> dict:
        """
        Retrieves visual/numeric chunks for `query` and returns pieces ready
        to merge with Task 1/2's text context before calling the LLM.
        """
        candidates = self.retriever.retrieve(query=query, k=self.retriever.top_k * 2)
        visual_docs = [
            d for d in candidates
            if d.metadata.get("content_type") in {"table", "chart", "diagram"}
            or d.metadata.get("_numeric_match")
        ][: self.retriever.top_k]

        if not visual_docs:
            return {
                "visual_context": "",
                "image_blocks": [],
                "citations": [],
                "has_visuals": False,
            }

        ctx = self.context_builder.build_context(documents=visual_docs, query=query)

        image_blocks = []
        for img in ctx["images"]:
            image_blocks.append({
                "type": "text",
                "text": f"Visual Evidence {img['reference']} | Source: {img['source']} | Page: {img['page']}",
            })
            image_blocks.append({
                "type": "image_url",
                "image_url": {"url": img["data_url"], "detail": "high"},
            })

        return {
            "visual_context": ctx["context"],
            "image_blocks": image_blocks,
            "citations": ctx["citations"],
            "has_visuals": True,
        }

    def as_runnable(self) -> RunnableLambda:
        """Returns an LCEL Runnable for `RunnableParallel` in Task 4."""
        return RunnableLambda(
            lambda x: self.get_visual_context(x["query"] if isinstance(x, dict) else str(x))
        )


__all__ = ["MultimodalVisualRetriever", "UnifiedRetriever", "ContextBuilder"]
