"""Optional Llama Guard 3 safety adapter (lazy-loaded).

Set ENABLE_LLAMA_GUARD=true to enable. Model weights are downloaded on first use;
keep this disabled in lightweight/local development environments.
"""
from __future__ import annotations

import os
from functools import lru_cache

from app.custom_exception import GuardrailTripException


@lru_cache(maxsize=1)
def _load_guard():
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformers and torch to enable Llama Guard.") from exc
    model_id = os.getenv("LLAMA_GUARD_MODEL", "meta-llama/Llama-Guard-3-1B")
    return pipeline("text-generation", model=model_id, tokenizer=model_id,
                    max_new_tokens=128, do_sample=False)


def _enabled() -> bool:
    return os.getenv("ENABLE_LLAMA_GUARD", "false").lower() in {"1", "true", "yes"}


def _is_unsafe(text: str) -> tuple[bool, str]:
    if not _enabled():
        return False, "disabled"
    guard = _load_guard()
    tokenizer = guard.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    result = guard(prompt, return_full_text=False)[0].get("generated_text", "")
    verdict = result.strip().splitlines()[0].lower() if result.strip() else ""
    return verdict.startswith("unsafe"), verdict or "unclassified"


def check_prompt_injection(text: str, threshold: float = 0.85) -> dict:
    """Check user input with Llama Guard when enabled; disabled is explicit."""
    unsafe, verdict = _is_unsafe(text)
    if unsafe:
        raise GuardrailTripException("Request blocked by Llama Guard safety policy.")
    return {"enabled": _enabled(), "verdict": verdict}


def filter_safe_chunks(chunks: list, threshold: float = 0.85) -> list:
    """Screen retrieved chunks; fail closed if every chunk is rejected."""
    if not _enabled():
        return chunks
    safe_chunks = []
    for chunk in chunks:
        unsafe, verdict = _is_unsafe(chunk.page_content[:4000])
        if not unsafe:
            safe_chunks.append(chunk)
    if chunks and not safe_chunks:
        raise GuardrailTripException("Retrieved documentation failed Llama Guard screening.")
    return safe_chunks
