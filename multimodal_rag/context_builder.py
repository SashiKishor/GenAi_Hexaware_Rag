"""
Multimodal Context & Prompt Builder for Northbridge Bank RAG.

Assembles retrieved text, table, chart, and diagram chunks into numbered evidence
blocks with page-level citations, conditionally attaches base64-encoded visual PNGs,
and constructs a compliance-grounded HumanMessage for Gemini generation.
"""


import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(
        self,
        max_images: int = 4,
        include_table_images: bool = False,
        image_max_bytes: int = 8 * 1024 * 1024,
    ):
        self.max_images = max_images
        self.include_table_images = include_table_images
        self.image_max_bytes = image_max_bytes

    @staticmethod
    def parse_numeric_facts(metadata: dict) -> list:
        facts = metadata.get("numeric_facts", "[]")
        if isinstance(facts, list):
            return facts
        try:
            return json.loads(facts)
        except (ValueError, TypeError):
            return []

    @staticmethod
    def get_source_reference(document: Document) -> dict:
        meta = document.metadata
        return {
            "source": meta.get("source", "unknown"),
            "version": meta.get("version", "unknown"),
            "page": meta.get("page", -1),
            "content_type": meta.get("content_type", "text"),
            "image_path": meta.get("image_path", ""),
        }

    @staticmethod
    def format_citation(document: Document, index: int) -> str:
        meta = document.metadata
        return (
            f"[{index}] Source: {meta.get('source', 'Unknown document')} | "
            f"Version: {meta.get('version', 'unknown')} | "
            f"Page: {meta.get('page', -1)} | "
            f"Type: {meta.get('content_type', 'text')}"
        )

    def get_visual_description(self, document: Document) -> str:
        meta = document.metadata
        caption = meta.get("caption") or meta.get("visual_caption", "")
        numeric_facts = self.parse_numeric_facts(meta)
        description = document.page_content

        if caption and caption not in description:
            description += f"\n\nVisual caption:\n{caption}"
        if numeric_facts:
            description += f"\n\nTranscribed numeric facts:\n{json.dumps(numeric_facts, ensure_ascii=False, indent=2)}"
        return description

    def load_image(self, image_path: str) -> dict | None:
        if not image_path:
            return None

        path = Path(image_path)
        if not path.exists():
            logger.warning("Image does not exist: %s", path)
            return None

        try:
            file_size = path.stat().st_size
            if file_size > self.image_max_bytes:
                logger.warning("Image exceeds configured size limit: %s", path)
                return None

            mime_type, _ = mimetypes.guess_type(str(path))
            if not mime_type or not mime_type.startswith("image/"):
                logger.warning("Unsupported image type: %s", path)
                return None

            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return {
                "path": str(path),
                "mime_type": mime_type,
                "data_url": f"data:{mime_type};base64,{encoded}",
                "size_bytes": file_size,
            }
        except OSError:
            logger.exception("Failed to load image: %s", path)
            return None

    def should_include_image(self, document: Document, query: str) -> bool:
        meta = document.metadata
        if not meta.get("image_path"):
            return False
        ctype = meta.get("content_type", "text")
        return ctype in {"chart", "diagram"} or (ctype == "table" and self.include_table_images)

    def build_context(self, documents: list[Document], query: str) -> dict:
        if not documents:
            raise ValueError("No retrieved documents were provided.")

        context_sections, citations, image_parts = [], [], []
        image_count = 0

        for idx, doc in enumerate(documents, start=1):
            meta = doc.metadata
            ctype = meta.get("content_type", "text")
            citation = self.format_citation(doc, idx)

            citations.append({"reference": f"[{idx}]", **self.get_source_reference(doc)})
            content = self.get_visual_description(doc) if ctype in {"chart", "diagram"} else doc.page_content
            context_sections.append(f"\n--- EVIDENCE {idx} ---\n{citation}\n\n{content}\n")

            if image_count < self.max_images and self.should_include_image(doc, query):
                if image_data := self.load_image(meta.get("image_path", "")):
                    image_parts.append({
                        "reference": f"[{idx}]",
                        "source": meta.get("source"),
                        "page": meta.get("page"),
                        "image_path": image_data["path"],
                        "mime_type": image_data["mime_type"],
                        "data_url": image_data["data_url"],
                    })
                    image_count += 1

        return {
            "query": query,
            "context": "\n".join(context_sections),
            "citations": citations,
            "images": image_parts,
            "image_count": image_count,
            "document_count": len(documents),
        }

    def build_message(self, query: str, context_data: dict) -> HumanMessage:
        prompt = (
            "You are Northbridge Bank's Banking and Financial Services RAG assistant.\n"
            "Answer the user's question using ONLY the supplied document evidence and images.\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"RETRIEVED DOCUMENT EVIDENCE:\n{context_data['context']}\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Treat the retrieved documents as your source of truth.\n"
            "2. For financial values, preserve exact amounts, currency, decimal places, and threshold boundaries.\n"
            "3. For tier questions, identify the correct range and explain why the supplied value belongs to it.\n"
            "4. Do not invent missing thresholds, fees, conditions, or account requirements.\n"
            "5. Distinguish current 2026 information from superseded 2025 information, and always state the effective date for any fee.\n"
            "6. If the documents do not establish an answer, explicitly say that the available evidence is insufficient.\n"
            "7. If images are supplied, inspect them and compare their visual information against the textual captions and extracted tables.\n"
            "8. Cite every important factual claim using the evidence references, such as [1] or [2].\n"
            "9. Never invent a source, page number, or citation.\n\n"
            "Provide a clear, concise answer."
        )

        message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in context_data["images"]:
            message_content.append({
                "type": "text",
                "text": f"Visual evidence {img['reference']} | Source: {img['source']} | Page: {img['page']}",
            })
            message_content.append({
                "type": "image_url",
                "image_url": {"url": img["data_url"], "detail": "high"},
            })

        return HumanMessage(content=message_content)

    def build(self, query: str, documents: list[Document]) -> dict:
        context_data = self.build_context(documents=documents, query=query)
        return {**context_data, "message": self.build_message(query=query, context_data=context_data)}
