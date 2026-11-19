"""
Vision-LLM Visual Captioner for Multimodal RAG Ingestion.

Uses Google Gemini with structured Pydantic outputs to transcribe text, exact numerical
values, tier thresholds, and step-by-step relationships from extracted tables, charts,
and process diagrams into searchable representations.
"""


import base64
import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()


# 1. Structured output models
class NumericFact(BaseModel):
    label: str = Field(description="What this numerical value represents")
    value: str = Field(description="Exact value transcribed from the image")
    unit: str = Field(description="Currency, percentage, quantity, etc.")
    condition: str = Field(description="Associated range or eligibility condition")


class VisualAnalysis(BaseModel):
    visual_type: str = Field(description="table, chart or diagram")
    caption: str = Field(description="Detailed description of the visual")
    transcribed_text: str = Field(description="Readable text and labels from the image")
    numeric_facts: List[NumericFact] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)


# 2. Visual Captioner
class VisualCaptioner:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Check your .env file.")

        self.model_name = model or os.getenv("GEMINI_VISION_MODEL", "gemini-3.1-flash-lite")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=0,
        )
        self.structured_llm = self.llm.with_structured_output(VisualAnalysis)

    def build_prompt(self, visual_type: str) -> str:
        return (
            f"You are a financial document extraction assistant for Northbridge Bank. "
            f"Analyze the provided {visual_type} accurately for a multimodal RAG system.\n\n"
            "Requirements:\n"
            "- Extract exact numbers, currency units, fee amounts, percentages, and balance thresholds.\n"
            "- Distinguish inclusive vs. exclusive bounds (<, <=, >, >=) and preserve row/column relationships.\n"
            "- For tier charts: extract tier name, min/max thresholds, boundary inclusivity, and balance measure.\n"
            "- For diagrams: explain the sequence of steps, decision branches, conditions, and exceptions.\n"
            "- Never invent, estimate, or assume missing values; report unreadable text in uncertainties."
        )

    def caption_image(self, image_path: str | Path, visual_type: str = "chart") -> dict:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image does not exist: {path}")

        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        ext = path.suffix.lower()
        if ext not in mime_map:
            raise ValueError(f"Unsupported image format: {ext}")

        b64_data = base64.b64encode(path.read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": self.build_prompt(visual_type)},
                {"type": "image_url", "image_url": f"data:{mime_map[ext]};base64,{b64_data}"},
            ]
        )

        result = self.structured_llm.invoke([message])
        output = result.model_dump()
        output["image_path"] = str(path.resolve())
        return output


if __name__ == "__main__":
    captioner = VisualCaptioner()
    result = captioner.caption_image("data/dummy/image.png", visual_type="chart")
    print(json.dumps(result, indent=2, ensure_ascii=False))
