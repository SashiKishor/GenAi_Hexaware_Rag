import asyncio
from datetime import date

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from rag.chain import (
    DEFAULT_NO_ANSWER,
    build_chain,
    detect_advice_category,
    generate_answer,
    stream_answer,
)
from rag.ranking_pipeline import RankedContext, build_ranking_pipeline
from rag.reranking import detect_injection  # noqa: F401  (import check only)


# ------------------------------------------------------------------ helpers
def doc(text="Business Plus monthly fee is Rs 650.", **meta):
    base = {"source": "Corpus_2026.md", "version": "3.0", "page": 7,
            "effective_date": "2026-01-01", "version_role": "current"}
    return Document(page_content=text, metadata={**base, **meta})


def ctx_of(*docs, flags=None, old=None):
    return RankedContext(question="q", docs=list(docs), flags=flags or [],
                         superseded_support=list(old or []))


def llm_returning(text):
    return FakeListChatModel(responses=[text])


def never_called():
    def boom(_):
        raise AssertionError("LLM must not be called")
    return RunnableLambda(boom)


def run(coro):
    return asyncio.run(coro)


def collect(agen):
    async def _go():
        return [event async for event in agen]
    return run(_go())


# ------------------------------------------------------------------ advice rule
@pytest.mark.parametrize("question, category", [
    ("Should I recommend this customer move savings into a fund?", "investment"),
    ("Should I move savings into a fund?", "investment"),
    ("Where should I invest my bonus?", "investment"),
    ("How can I reduce my tax bill this year?", "tax"),
    ("Will my loan be approved?", "credit"),
    ("Should the bank approve this application?", "credit"),
])
def test_advice_questions_are_detected(question, category):
    assert detect_advice_category(question) == category


@pytest.mark.parametrize("question", [
    "What is the monthly fee for the Business Plus account and what waives it?",
    "List the KYC documents required to open a sole-trader account.",
    "Which tier applies at an average balance of 45,000?",
    "Can I transfer funds between accounts?",
    "What documents are needed for loan approval?",
    "Does the 2024 version of the overdraft policy still apply?",
])
def test_normal_questions_are_not_blocked(question):
    assert detect_advice_category(question) is None


def test_advice_refusal_names_a_human_channel_and_skips_llm():
    result = run(generate_answer("Should I move savings into a fund?",
                                 ctx_of(doc()), never_called()))
    assert "Advisory Desk" in result["answer"]
    assert result["citations"] == [] and "advice_refused" in result["flags"]
    assert "no_llm_call" in result["flags"]


# ------------------------------------------------------------------ empty / failed retrieval
@pytest.mark.parametrize("flags, expected", [
    ([], DEFAULT_NO_ANSWER),
    (["not_found"], DEFAULT_NO_ANSWER),
    (["retrieval_error"], "could not search"),
    (["low_confidence"], "confident match"),
    (["version_unresolved"], "cannot confirm which version"),
])
def test_no_documents_means_refusal_without_llm(flags, expected):
    result = run(generate_answer("What is the fee?", ctx_of(flags=flags), never_called()))
    assert expected in result["answer"]
    assert result["citations"] == [] and "no_llm_call" in result["flags"]


def test_plain_list_input_is_supported():
    result = run(generate_answer("What is the fee?", [], never_called()))
    assert result["answer"] == DEFAULT_NO_ANSWER


# ------------------------------------------------------------------ normal answer
def test_answer_has_citation_chunks_and_effective_date():
    llm = llm_returning("The Business Plus monthly fee is Rs 650 (effective 1 January 2026) [S1].")
    result = run(generate_answer("What is the Business Plus fee?", ctx_of(doc()), llm))

    assert result["citations"][0]["source_id"] == "S1"
    assert result["citations"][0]["document_id"] == "Corpus_2026.md"
    assert result["citations"][0]["page_or_section"] == "7"
    assert result["citations"][0]["status"] == "Current"
    assert result["effective_date"] == "2026-01-01"
    assert result["is_superseded_warning"] is False
    assert len(result["chunks"]) == 1
    assert "effective_date_appended" not in result["flags"]  # the model already gave it


def test_effective_date_is_added_when_model_forgets_it():
    llm = llm_returning("The Business Plus monthly fee is Rs 650 [S1].")
    result = run(generate_answer("fee?", ctx_of(doc()), llm))
    assert "effective date: 2026-01-01" in result["answer"]
    assert "effective_date_appended" in result["flags"]


def test_missing_effective_date_is_stated_not_invented():
    llm = llm_returning("The fee is Rs 650 [S1].")
    result = run(generate_answer("fee?", ctx_of(doc(effective_date="unknown")), llm))
    assert "not available in the source document" in result["answer"]
    assert result["effective_date"] is None


def test_non_fee_answer_gets_no_date_footer():
    llm = llm_returning("Provide proprietor identity and address evidence [S1].")
    result = run(generate_answer("KYC documents?", ctx_of(doc()), llm))
    assert "Effective date:" not in result["answer"]


def test_answer_without_citation_is_flagged_and_invented_id_is_flagged():
    r1 = run(generate_answer("q", ctx_of(doc()), llm_returning("No source here.")))
    assert "no_inline_citation" in r1["flags"] and r1["citations"]
    r2 = run(generate_answer("q", ctx_of(doc()), llm_returning("Made up [S9].")))
    assert "unknown_citation_id" in r2["flags"]


def test_two_sources_with_different_dates_have_no_single_date():
    d1, d2 = doc(effective_date="2026-01-01"), doc(effective_date="2025-01-01", version="2.0")
    result = run(generate_answer("q", ctx_of(d1, d2), llm_returning("A [S1] and B [S2].")))
    assert result["effective_date"] is None
    assert [c["effective_date"] for c in result["citations"]] == ["2026-01-01", "2025-01-01"]


# ------------------------------------------------------------------ superseded
def test_superseded_only_is_used_with_a_warning():
    old = doc(version="2.0", effective_date="2025-01-01", version_role=None)
    ctx = ctx_of(old=[old], flags=["superseded_only"])
    result = run(generate_answer("What was the fee?", ctx,
                                 llm_returning("The fee was Rs 575 (effective 2025-01-01) [S1].")))
    assert result["answer"].startswith("WARNING")
    assert result["is_superseded_warning"] is True
    assert result["citations"][0]["status"] == "Superseded"


def test_end_to_end_with_real_ranking_pipeline():
    """Real VersionCatalog on the real corpus; only the retriever is fake."""
    old = Document(page_content="Business Plus monthly fee is Rs 575.",
                   metadata={"source": "Northbridge_Bank_Synthetic_Corpus_2025.md", "page": 3})
    new = Document(page_content="Business Plus monthly fee is Rs 650.",
                   metadata={"source": "Northbridge_Bank_Synthetic_Corpus_2026.md", "page": 3})

    class FakeRetriever:
        def __init__(self, docs):
            self.docs = docs

        def search(self, query, k=None, **kwargs):
            return self.docs

    def ask(docs, question):
        pipeline = build_ranking_pipeline(FakeRetriever(docs), use_cross_encoder=False,
                                          today_provider=lambda: date(2026, 11, 19))
        return pipeline.ask(question)

    current = ask([old, new], "What is the Business Plus monthly fee?")
    result = run(generate_answer("What is the Business Plus monthly fee?", current,
                                 llm_returning("The fee is Rs 650 [S1].")))
    assert result["is_superseded_warning"] is False
    assert result["effective_date"] == "2026-01-01"

    only_old = ask([old], "What is the Business Plus monthly fee?")
    assert "superseded_only" in only_old.flags
    result = run(generate_answer("What is the Business Plus monthly fee?", only_old,
                                 llm_returning("The fee is Rs 575 [S1].")))
    assert result["is_superseded_warning"] is True
    assert result["answer"].startswith("WARNING")


# ------------------------------------------------------------------ prompt wiring (LCEL)
def test_chain_puts_sources_and_question_in_the_prompt():
    seen = {}

    def capture(prompt_value):
        seen["text"] = prompt_value.to_string()
        return "ok [S1]"

    chain = build_chain(RunnableLambda(capture))
    out = chain.invoke({"question": "What is the fee?",
                        "chunks": [Document(page_content="Fee Rs 650",
                                            metadata={"source_id": "S1", "source": "A.md"})]})
    assert out == "ok [S1]"
    assert "Source ID: [S1]" in seen["text"] and "Fee Rs 650" in seen["text"]
    assert "What is the fee?" in seen["text"]


# ------------------------------------------------------------------ timeout / errors
def test_timeout_raises_asyncio_timeout_error():
    async def slow(_):
        await asyncio.sleep(5)
        return "late"

    with pytest.raises(asyncio.TimeoutError):
        run(generate_answer("q", ctx_of(doc()), RunnableLambda(slow), timeout_seconds=0.05))


def test_llm_failure_is_propagated_without_leaking_content():
    def fail(_):
        raise RuntimeError("secret account 12345678901")

    with pytest.raises(RuntimeError):
        run(generate_answer("q", ctx_of(doc()), RunnableLambda(fail)))


def test_empty_generation_becomes_a_refusal():
    result = run(generate_answer("q", ctx_of(doc()), llm_returning("   ")))
    assert result["answer"] == DEFAULT_NO_ANSWER and "empty_generation" in result["flags"]


# ------------------------------------------------------------------ streaming
def test_stream_yields_tokens_then_one_final_event():
    text = "The fee is Rs 650 [S1]."
    events = collect(stream_answer("fee?", ctx_of(doc()), llm_returning(text)))

    tokens = [e["data"] for e in events if e["type"] == "token"]
    assert len(tokens) > 3                                   # really streamed
    assert events[-1]["type"] == "final"
    assert sum(e["type"] == "final" for e in events) == 1
    final = events[-1]["data"]
    assert "".join(tokens) == final["answer"]                # what was streamed == final answer
    assert "effective date: 2026-01-01" in final["answer"]   # footer streamed too


def test_stream_refusal_has_no_llm_call():
    events = collect(stream_answer("Should I move savings into a fund?",
                                   ctx_of(doc()), never_called()))
    assert [e["type"] for e in events] == ["token", "final"]
    assert "no_llm_call" in events[-1]["data"]["flags"]


def test_stream_timeout_ends_with_error_event():
    async def slow(_):
        await asyncio.sleep(5)
        return "late"

    events = collect(stream_answer("q", ctx_of(doc()), RunnableLambda(slow), timeout_seconds=0.05))
    assert events[-1] == {"type": "error", "code": "TIMEOUT",
                          "message": "The request timed out. Please try again."}


def test_stream_llm_error_ends_with_error_event():
    def fail(_):
        raise RuntimeError("boom")

    events = collect(stream_answer("q", ctx_of(doc()), RunnableLambda(fail)))
    assert events[-1]["type"] == "error" and events[-1]["code"] == "LLM_ERROR"