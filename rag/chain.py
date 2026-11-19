
import asyncio
import logging
import json
import re
from datetime import date
from typing import Any, AsyncIterator, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag.prompt import create_prompt, format_context

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30

FLAG_NOT_FOUND = "not_found"
FLAG_SUPERSEDED_ONLY = "superseded_only"

FLAG_ADVICE_REFUSED = "advice_refused"
FLAG_NO_LLM_CALL = "no_llm_call"
FLAG_NO_CITATION = "no_inline_citation"
FLAG_UNKNOWN_CITATION = "unknown_citation_id"
FLAG_DATE_APPENDED = "effective_date_appended"
FLAG_EMPTY_GENERATION = "empty_generation"

ADVICE_REFUSAL_CHANNELS = {
    "investment": (
        "I cannot provide investment advice or fund recommendations. "
        "Please refer this customer to the Northbridge Wealth & Investment "
        "Advisory Desk (ext. 4410)."
    ),
    "tax": (
        "I cannot provide tax advice. Please refer this customer to "
        "Northbridge Tax & Fiduciary Advisory Services (ext. 4425)."
    ),
    "credit": (
        "I cannot make credit-approval or underwriting decisions. Please "
        "submit the application to the Northbridge Credit Underwriting "
        "Team (ext. 4450)."
    ),
}

_ASSETS = (
    r"(?:funds?|stocks?|shares|bonds?|etfs?|mutual|crypto\w*|"
    r"securities|investments?)"
)

ADVICE_PATTERNS = {
    "investment": [
        r"\binvestment advice\b",
        r"\bshould (?:i|we|they|he|she|the customer)\b.{0,20}\binvest\b",
        r"\bwhere (?:should|can|do) (?:i|we|they)\b.{0,20}\binvest\b",
        r"\b(?:move|put|shift|switch|invest)\b.{0,40}\b(?:into|in)\b.{0,20}\b" + _ASSETS,
        r"\brecommend\w*\b.{0,60}\b" + _ASSETS,
        r"\b(?:which|what|best)\b.{0,30}\b(?:mutual funds?|stocks?|etfs?)\b.{0,30}\b(?:buy|pick|choose|best)\b",
    ],
    "tax": [
        r"\btax (?:advice|planning|strategy|saving|savings|benefits?|implications?|consequences?)\b",
        r"\b(?:reduce|avoid|minimi[sz]e|lower)\b.{0,20}\btax(?:es)?\b",
        r"\bhow (?:should|do|can) (?:i|we|they|the customer)\b.{0,15}\b(?:file|declare|report)\b.{0,20}\btax",
        r"\bwhich tax option\b",
    ],
    "credit": [
        r"\bcredit[- ]approval advice\b",
        r"\b(?:will|would|can|could)\b.{0,30}\b(?:loan|credit|mortgage|overdraft|application|facility)\b"
        r".{0,20}\b(?:be )?(?:approved|sanctioned|accepted|rejected|declined)\b",
        r"\bshould (?:we|i|the bank|you)\b.{0,15}\b(?:approve|sanction|lend|reject|decline|grant)\b",
        r"\b(?:approve|sanction|decline|reject)\b.{0,20}\b(?:this|his|her|their|my|the)\b"
        r".{0,20}\b(?:loan|credit|application|overdraft|facility|mortgage)\b",
        r"\bis (?:this customer|he|she|the applicant|the customer)\b.{0,25}\b(?:creditworthy|credit worthy)\b",
        r"\bwill i get\b.{0,30}\b(?:loan|credit|overdraft)\b",
    ],
}

_ADVICE_REGEX = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in ADVICE_PATTERNS.items()
}


def detect_advice_category(question: str) -> Optional[str]:
    for category, regexes in _ADVICE_REGEX.items():
        if any(rx.search(question) for rx in regexes):
            return category
    return None


def is_advice_request(question: str) -> bool:
    return detect_advice_category(question) is not None


DEFAULT_NO_ANSWER = (
    "I cannot answer this based on Northbridge Bank's verified documentation."
)

NO_ANSWER_MESSAGES = [
    ("empty_question", "Please enter a product, fee or compliance question."),
    ("retrieval_error", "I could not search the documents right now. Please try again shortly."),
    ("ranking_error", "I could not search the documents right now. Please try again shortly."),
    ("version_unresolved", (
        "I found related documents, but I cannot confirm which version is current, "
        "so I cannot give a reliable answer. Please check the latest circular."
    )),
    ("low_confidence", (
        "I could not find a confident match in Northbridge Bank's documentation. "
        "Please rephrase the question or check the source document."
    )),
]

SUPERSEDED_NOTICE = (
    "WARNING: Only a SUPERSEDED (outdated) document supports this answer. "
    "Do not treat it as current policy; confirm with the latest version."
)


def _no_answer_message(flags: list[str]) -> str:
    for flag, message in NO_ANSWER_MESSAGES:
        if flag in flags:
            return message
    return DEFAULT_NO_ANSWER


def _unpack(ctx: Any) -> tuple[list[Document], list[str], bool]:
    if not hasattr(ctx, "docs"):
        docs = [getattr(item, "document", item) for item in (ctx or [])]
        return docs, [], False

    docs = list(ctx.docs)
    flags = list(ctx.flags)
    old = list(getattr(ctx, "superseded_support", []))

    if not docs and old and FLAG_SUPERSEDED_ONLY in flags:
        return old, flags, True

    return docs, flags, False


def _prepare_docs(docs: list[Document], superseded: bool) -> list[Document]:
    prepared = []

    for i, doc in enumerate(docs, start=1):
        meta = dict(doc.metadata)
        meta["source_id"] = f"S{i}"

        role = meta.get("version_role")
        status_value = str(
            meta.get("status", meta.get("version_status", ""))
        ).lower()

        if not role and status_value in {"current", "superseded"}:
            role = "current" if status_value == "current" else "historical"

        if superseded or role == "historical":
            meta["status"] = "superseded"
            meta["version_status"] = "superseded"
        elif role == "current":
            meta["status"] = "current"
            meta["version_status"] = "current"

        prepared.append(
            Document(
                page_content=doc.page_content,
                metadata=meta,
            )
        )

    return prepared


def build_chain(llm):
    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: format_context(x["chunks"])
            )
        )
        | create_prompt()
        | llm
        | StrOutputParser()
    )


# Matches source IDs both in [S1] and [S1, S3].
_CITATION_RE = re.compile(r"\bS(\d+)\b")

_FEE_RE = re.compile(
    r"(?:₹|\brs\.?|\binr\b|\$|£|€)\s?\d|\d\s?%",
    re.IGNORECASE,
)


def _parse_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _date_spellings(d: date) -> list[str]:
    return [
        d.isoformat(),
        f"{d.day} {d:%B %Y}",
        f"{d.day:02d} {d:%B %Y}",
        f"{d:%B} {d.day}, {d.year}",
        f"{d.day} {d:%b %Y}",
    ]


def _chunk_info(doc: Document) -> dict[str, Any]:
    m = doc.metadata

    section = m.get("section") or ""

    if not section and m.get("headings"):
        try:
            parsed = json.loads(m["headings"])
            section = (
                " > ".join(parsed)
                if isinstance(parsed, list)
                else str(parsed)
            )
        except Exception:
            section = str(m["headings"])

    page = m.get("page")
    page_str = (
        f"Page {page}"
        if page not in (None, "", -1)
        else "Unknown page"
    )

    location = (
        f"{page_str} — {section}"
        if section
        else page_str
    )

    return {
        "source_id": m["source_id"],
        "document_id": str(m.get("source", "Unknown document")),
        "version": str(m.get("version", "Unknown")),
        "page_or_section": location,
        "section": section or "General",
        "status": str(m.get("status", "Unknown")).capitalize(),
        "effective_date": str(
            m.get("effective_date") or "Not available"
        ),
        "content_type": str(m.get("content_type") or "text"),
        "image_path": str(m.get("image_path") or ""),
    }


def _guard_chunk_info(doc: Document) -> dict[str, Any]:
    """
    Evidence payload for the output guard.

    Unlike API citation metadata, this includes the actual retrieved text.
    The output guard must redact it before sending it to the audit model.
    """
    citation_info = _chunk_info(doc)

    return {
        **citation_info,
        "page": doc.metadata.get("page", "N/A"),
        "content": doc.page_content,
    }


def _effective_date_footer(
    answer: str,
    docs: list[Document],
) -> str:
    if not _FEE_RE.search(answer):
        return ""

    cited = {
        f"S{number}"
        for number in _CITATION_RE.findall(answer)
    }

    used = [
        doc for doc in docs
        if doc.metadata["source_id"] in cited
    ] or docs

    lines = []

    for doc in used:
        source_id = doc.metadata["source_id"]
        effective_date = _parse_date(
            doc.metadata.get("effective_date")
        )

        if effective_date is None:
            lines.append(
                f"[{source_id}] effective date: "
                "not available in the source document"
            )
        elif not any(
            spelling.lower() in answer.lower()
            for spelling in _date_spellings(effective_date)
        ):
            lines.append(
                f"[{source_id}] effective date: "
                f"{effective_date.isoformat()}"
            )

    return (
        "\n\nEffective date: " + "; ".join(lines)
        if lines
        else ""
    )


def _top_level_effective_date(
    docs: list[Document],
) -> Optional[str]:
    dates = {
        str(doc.metadata["effective_date"])
        for doc in docs
        if _parse_date(doc.metadata.get("effective_date"))
    }

    return dates.pop() if len(dates) == 1 else None


def _build_result(
    answer: str,
    docs: list[Document],
    flags: list[str],
    superseded: bool,
) -> dict[str, Any]:

    flags = list(flags)

    chunks = [_chunk_info(doc) for doc in docs]
    known = {chunk["source_id"] for chunk in chunks}

    cited = {
        f"S{number}"
        for number in _CITATION_RE.findall(answer)
    }

    if not cited:
        flags.append(FLAG_NO_CITATION)

    unknown = cited - known

    if unknown:
        flags.append(FLAG_UNKNOWN_CITATION)
        logger.warning(
            "Chain generated unknown citation IDs: %s",
            sorted(unknown),
        )

    citations = [
        chunk for chunk in chunks
        if chunk["source_id"] in cited
    ]

    if not citations and not cited:
        citations = chunks

    return {
        "answer": answer,
        "citations": citations,
        "chunks": chunks,

        # Internal evidence payload. Remove before sending API response.
        "guard_context": [
            _guard_chunk_info(doc)
            for doc in docs
        ],

        "flags": list(dict.fromkeys(flags)),
        "effective_date": _top_level_effective_date(docs),
        "is_superseded_warning": superseded,
    }


def _refusal_result(
    message: str,
    flags: list[str],
) -> dict[str, Any]:
    return {
        "answer": message,
        "citations": [],
        "chunks": [],
        "guard_context": [],
        "flags": list(
            dict.fromkeys([*flags, FLAG_NO_LLM_CALL])
        ),
        "effective_date": None,
        "is_superseded_warning": False,
    }


def _pre_llm_refusal(
    question: str,
    docs: list[Document],
    flags: list[str],
) -> Optional[dict[str, Any]]:

    category = detect_advice_category(question)

    if category:
        return _refusal_result(
            ADVICE_REFUSAL_CHANNELS[category],
            [*flags, FLAG_ADVICE_REFUSED],
        )

    if not docs:
        return _refusal_result(
            _no_answer_message(flags),
            [*flags, FLAG_NOT_FOUND],
        )

    return None


async def generate_answer(
    question: str,
    ctx: Any,
    llm,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:

    docs, flags, superseded = _unpack(ctx)

    refusal = _pre_llm_refusal(
        question,
        docs,
        flags,
    )

    if refusal:
        logger.info(
            "Chain: refused without LLM call flags=%s",
            refusal["flags"],
        )
        return refusal

    docs = _prepare_docs(docs, superseded)
    chain = build_chain(llm)

    try:
        body = await asyncio.wait_for(
            chain.ainvoke(
                {
                    "question": question,
                    "chunks": docs,
                }
            ),
            timeout=timeout_seconds,
        )

    except asyncio.TimeoutError:
        logger.error(
            "Chain: LLM timed out after %ss",
            timeout_seconds,
        )
        raise

    except Exception as exc:
        logger.error(
            "Chain: LLM call failed: %s",
            type(exc).__name__,
        )
        raise

    body = body.strip()

    if not body:
        return _refusal_result(
            DEFAULT_NO_ANSWER,
            [*flags, FLAG_EMPTY_GENERATION],
        )

    footer = _effective_date_footer(body, docs)

    if footer:
        flags.append(FLAG_DATE_APPENDED)

    prefix = (
        f"{SUPERSEDED_NOTICE}\n\n"
        if superseded
        else ""
    )

    result = _build_result(
        prefix + body + footer,
        docs,
        flags,
        superseded,
    )

    logger.info(
        "Chain: answered sources=%d flags=%s",
        len(docs),
        result["flags"],
    )

    return result


async def stream_answer(
    question: str,
    ctx: Any,
    llm,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> AsyncIterator[dict[str, Any]]:

    docs, flags, superseded = _unpack(ctx)

    refusal = _pre_llm_refusal(
        question,
        docs,
        flags,
    )

    if refusal:
        logger.info(
            "Chain: refused without LLM call flags=%s",
            refusal["flags"],
        )
        yield {
            "type": "token",
            "data": refusal["answer"],
        }
        yield {
            "type": "final",
            "data": refusal,
        }
        return

    docs = _prepare_docs(docs, superseded)
    chain = build_chain(llm)

    prefix = (
        f"{SUPERSEDED_NOTICE}\n\n"
        if superseded
        else ""
    )

    if prefix:
        yield {
            "type": "token",
            "data": prefix,
        }

    pieces: list[str] = []

    try:
        async with asyncio.timeout(timeout_seconds):
            async for token in chain.astream(
                {
                    "question": question,
                    "chunks": docs,
                }
            ):
                pieces.append(token)
                yield {
                    "type": "token",
                    "data": token,
                }

    except TimeoutError:
        logger.error(
            "Chain: stream timed out after %ss",
            timeout_seconds,
        )
        yield {
            "type": "error",
            "code": "TIMEOUT",
            "message": "The request timed out. Please try again.",
        }
        return

    except Exception as exc:
        logger.error(
            "Chain: stream failed: %s",
            type(exc).__name__,
        )
        yield {
            "type": "error",
            "code": "LLM_ERROR",
            "message": (
                "Something went wrong while generating the answer."
            ),
        }
        return

    body = "".join(pieces).strip()

    if not body:
        refusal_result = _refusal_result(
            DEFAULT_NO_ANSWER,
            [*flags, FLAG_EMPTY_GENERATION],
        )

        yield {
            "type": "final",
            "data": refusal_result,
        }
        return

    footer = _effective_date_footer(body, docs)

    if footer:
        flags.append(FLAG_DATE_APPENDED)
        yield {
            "type": "token",
            "data": footer,
        }

    final_result = _build_result(
        prefix + body + footer,
        docs,
        flags,
        superseded,
    )

    yield {
        "type": "final",
        "data": final_result,
    }
