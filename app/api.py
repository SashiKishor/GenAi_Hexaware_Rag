
import os
import jwt
import uuid
import json
import time
import asyncio
import logging
import re

from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse

from langchain_core.documents import Document

from app.custom_exception import GuardrailTripException
from app.models import User, Token, TokenData, AskRequest, AskResponse
from app.jwt_utils import verify_password, create_access_token, decode_access_token
from app.database import get_user_by_username, create_user

from security.Pii_Redactor import PIIRedactor, pii_redactor
from security.input_guard import run_input_guardrail
from security.output_guard import run_output_guardrail
from security.llam import check_prompt_injection, filter_safe_chunks

from rag.chain import generate_answer, stream_answer
from rag.reranking import detect_injection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


GEMINI_CHAT_MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    "gemini-3.5-flash-lite",
)

RETRIEVAL_TIMEOUT = get_env_float(
    "RETRIEVAL_TIMEOUT_SECONDS",
    30,
)

LLM_TIMEOUT = get_env_float(
    "LLM_TIMEOUT_SECONDS",
    45,
)

GUARDRAIL_TIMEOUT = get_env_float(
    "GUARDRAIL_TIMEOUT_SECONDS",
    30,
)


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


app = FastAPI(
    title="Northbridge Bank Multimodal RAG API",
    version="1.0.0",
)


@app.exception_handler(GuardrailTripException)
async def guardrail_handler(
    request: Request,
    exc: GuardrailTripException,
):
    logger.warning("Guardrail blocked a request.")

    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error_code": "GUARDRAIL_TRIP",
            "message": str(exc),
        },
    )


@app.exception_handler(asyncio.TimeoutError)
async def timeout_handler(
    request: Request,
    exc: asyncio.TimeoutError,
):
    logger.error("Request timed out.")

    return JSONResponse(
        status_code=504,
        content={
            "success": False,
            "error_code": "TIMEOUT",
            "message": "Request timed out. Please try again.",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled API error: %s",
        type(exc).__name__,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "message": "Something went wrong. Please contact support.",
        },
    )


@lru_cache()
def get_redactor() -> PIIRedactor:
    return pii_redactor


@lru_cache()
def get_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        logger.error("GOOGLE_API_KEY is missing.")
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info(
            "Initializing main Gemini model: %s",
            GEMINI_CHAT_MODEL,
        )

        return ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            google_api_key=google_api_key,
            temperature=0,
            timeout=LLM_TIMEOUT,
            max_retries=1,
        )

    except Exception as exc:
        logger.error(
            "Gemini initialization failed: %s",
            type(exc).__name__,
        )
        return None


@lru_cache()
def get_ranking_pipeline():
    try:
        from vector_store.chroma_db import ChromaDB
        from rag.retrieval import HybridRetriever
        from rag.ranking_pipeline import (
            build_ranking_pipeline,
            RankingConfig,
        )

        db = ChromaDB()
        retriever = HybridRetriever(
            db.get_vector_store()
        )

        floor = get_env_float(
            "MIN_RELEVANCE",
            0.15,
        )

        config = RankingConfig(
            min_relevance=floor,
        )

        logger.info("Initializing ranking pipeline.")

        return build_ranking_pipeline(
            retriever,
            config=config,
        )

    except Exception as exc:
        logger.exception(
            "Ranking pipeline initialization failed: %s",
            type(exc).__name__,
        )
        return None


@lru_cache()
def get_northbridge_pipeline():
    ranking = get_ranking_pipeline()
    llm = get_llm()

    if ranking is None or llm is None:
        return None

    try:
        from rag.full_pipeline import NorthbridgeRAGPipeline

        logger.info(
            "Initializing multimodal RAG pipeline."
        )

        return NorthbridgeRAGPipeline(
            ranking,
            llm,
        )

    except Exception as exc:
        logger.exception(
            "Multimodal pipeline initialization failed: %s",
            type(exc).__name__,
        )
        return None


def _redact_retrieved_context(
    ctx,
    redactor: PIIRedactor,
):
    if ctx is None or not hasattr(ctx, "docs"):
        return ctx

    ctx.docs = [
        Document(
            page_content=redactor.redact(
                doc.page_content
            ),
            metadata=dict(doc.metadata),
        )
        for doc in ctx.docs
    ]

    ctx.superseded_support = [
        Document(
            page_content=redactor.redact(
                doc.page_content
            ),
            metadata=dict(doc.metadata),
        )
        for doc in getattr(
            ctx,
            "superseded_support",
            [],
        )
    ]

    return ctx


def _guard_chunks(ctx):
    if ctx is None or not hasattr(ctx, "docs"):
        return ctx

    ctx.docs = [
        doc
        for doc in ctx.docs
        if not detect_injection(
            doc.page_content
        )
    ]

    ctx.superseded_support = [
        doc
        for doc in getattr(
            ctx,
            "superseded_support",
            [],
        )
        if not detect_injection(
            doc.page_content
        )
    ]

    ctx.docs = filter_safe_chunks(
        ctx.docs
    )

    ctx.superseded_support = filter_safe_chunks(
        ctx.superseded_support
    )

    return ctx


def _get_guard_context(result: dict) -> list:
    guard_context = result.get(
        "guard_context",
        [],
    )

    if not isinstance(guard_context, list):
        logger.error(
            "Invalid guard_context returned by RAG chain."
        )
        return []

    valid_context = [
        chunk
        for chunk in guard_context
        if isinstance(chunk, dict)
        and chunk.get("source_id")
        and chunk.get("content")
    ]

    logger.info(
        "GUARD CONTEXT SOURCE IDS: %s",
        [
            chunk.get("source_id")
            for chunk in valid_context
        ],
    )

    citations = result.get(
        "citations",
        [],
    )

    logger.info(
        "GUARD CITATION SOURCE IDS: %s",
        [
            citation.get("source_id")
            for citation in citations
            if isinstance(citation, dict)
        ],
    )

    if not valid_context:
        logger.error(
            "No valid guard context was returned by the RAG chain."
        )

    return valid_context


def _log_citation_debug(
    answer: str,
    citations: list,
):
    answer_source_ids = sorted(
        set(
            re.findall(
                r"\[S(\d+(?:\s*,\s*S\d+)*)\]",
                answer,
            )
        )
    )

    extracted_ids = set()

    for group in answer_source_ids:
        extracted_ids.update(
            re.findall(
                r"S(\d+)",
                group,
            )
        )

    answer_source_ids = sorted(
        extracted_ids,
        key=int,
    )

    returned_source_ids = [
        str(citation.get("source_id"))
        for citation in citations
        if isinstance(citation, dict)
        and citation.get("source_id") is not None
    ]

    logger.info(
        "ANSWER SOURCE IDS: %s",
        answer_source_ids,
    )

    logger.info(
        "RETURNED SOURCE IDS: %s",
        returned_source_ids,
    )

    missing_ids = [
        source_id
        for source_id in answer_source_ids
        if f"S{source_id}" not in returned_source_ids
    ]

    if missing_ids:
        logger.warning(
            "Answer contains source markers without matching citation metadata: %s",
            missing_ids,
        )


def _sse(event: dict) -> str:
    return (
        "data: "
        + json.dumps(
            event,
            ensure_ascii=False,
            default=str,
        )
        + "\n\n"
    )


def _log_elapsed(
    stage: str,
    start_time: float,
):
    elapsed = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    logger.info(
        "%s completed in %.2f ms",
        stage,
        elapsed,
    )

    return elapsed


security = HTTPBearer(
    scheme_name="HTTPBearer",
    description="Enter your JWT access token.",
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    token = credentials.credentials

    try:
        payload = decode_access_token(token)
        username = payload.get("sub")

        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload is missing user subject.",
                headers={
                    "WWW-Authenticate": "Bearer",
                },
            )

        return TokenData(
            username=username,
            role=payload.get("role"),
            session_id=payload.get("session_id"),
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )


async def require_admin(
    current_user: TokenData = Depends(get_current_user),
) -> TokenData:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )

    return current_user


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Northbridge Bank Multimodal RAG API",
    }


@app.post(
    "/login",
    response_model=Token,
    summary="User Login",
    tags=["Authentication"],
)
async def login(user_credentials: User):
    db_user = get_user_by_username(
        user_credentials.username
    )

    if (
        not db_user
        or not verify_password(
            user_credentials.password,
            db_user["hashed_password"],
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    session_id = str(uuid.uuid4())

    access_token = create_access_token(
        data={
            "sub": db_user["username"],
            "role": db_user.get(
                "role",
                "user",
            ),
            "session_id": session_id,
        }
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        session_id=session_id,
    )


@app.post(
    "/register",
    summary="Register New User",
    tags=["Authentication"],
    status_code=status.HTTP_201_CREATED,
)
async def register(user_credentials: User):
    existing_user = get_user_by_username(
        user_credentials.username
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already registered.",
        )

    create_user(
        username=user_credentials.username,
        plain_password=user_credentials.password,
        role="user",
    )

    return {
        "message": "User registered successfully.",
    }


@app.post(
    "/registerManager",
    summary="Register Manager",
    tags=["Authentication"],
    status_code=status.HTTP_201_CREATED,
)
async def register_manager(
    user_credentials: User,
    admin_user: TokenData = Depends(require_admin),
):
    existing_user = get_user_by_username(
        user_credentials.username
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already registered.",
        )

    create_user(
        username=user_credentials.username,
        plain_password=user_credentials.password,
        role="manager",
    )

    return {
        "message": "Manager registered successfully.",
    }


@app.get(
    "/sessionId",
    tags=["Protected"],
)
async def get_session_id(
    current_user: TokenData = Depends(get_current_user),
):
    return {
        "session_id": current_user.session_id,
        "username": current_user.username,
        "role": current_user.role,
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask Question (RAG)",
    tags=["RAG"],
)
async def ask_question(
    request: AskRequest,
    current_user: TokenData = Depends(get_current_user),
    redactor: PIIRedactor = Depends(get_redactor),
    llm=Depends(get_llm),
    ranking_pipeline=Depends(get_ranking_pipeline),
):
    request_start = time.perf_counter()

    safe_query = redactor.redact(
        request.query
    )

    if not llm or not ranking_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "RAG_UNAVAILABLE",
                "message": (
                    "The retrieval or language model service "
                    "is unavailable."
                ),
            },
        )

    try:
        stage_start = time.perf_counter()

        input_result = await asyncio.wait_for(
            run_input_guardrail(safe_query),
            timeout=GUARDRAIL_TIMEOUT,
        )

        _log_elapsed(
            "Input guardrail",
            stage_start,
        )

        logger.info(
            "Input guardrail timings: %s",
            input_result.latencies_ms,
        )

        if not input_result.allowed:
            return AskResponse(
                answer=(
                    input_result.refusal_message
                    or "Request blocked by policy."
                ),
                citations=[],
                effective_date=None,
                is_superseded_warning=False,
            )

        safe_query = input_result.sanitized_text

        if detect_injection(safe_query):
            raise GuardrailTripException(
                "Request blocked: prompt-injection pattern detected."
            )

        check_prompt_injection(safe_query)

        pipeline = get_northbridge_pipeline()

        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Multimodal RAG pipeline unavailable.",
            )

        stage_start = time.perf_counter()

        ctx = await asyncio.wait_for(
            asyncio.to_thread(
                pipeline._retrieve_combined,
                safe_query,
            ),
            timeout=RETRIEVAL_TIMEOUT,
        )

        _log_elapsed(
            "Combined retrieval",
            stage_start,
        )

        ctx = _guard_chunks(ctx)

        ctx = _redact_retrieved_context(
            ctx,
            redactor,
        )

        if not ctx or not getattr(ctx, "docs", None):
            return AskResponse(
                answer=(
                    "I cannot answer this question because "
                    "no supporting Northbridge Bank document "
                    "was found."
                ),
                citations=[],
                effective_date=None,
                is_superseded_warning=False,
            )

        stage_start = time.perf_counter()

        result = await generate_answer(
            question=safe_query,
            ctx=ctx,
            llm=llm,
            timeout_seconds=LLM_TIMEOUT,
        )

        answer_text = result.get(
            "answer",
            "",
        )

        citations = result.get(
            "citations",
            [],
        )

        logger.info(
            "GENERATED ANSWER: %s",
            answer_text,
        )

        _log_citation_debug(
            answer_text,
            citations,
        )

        _log_elapsed(
            "Answer generation",
            stage_start,
        )

        guard_context = _get_guard_context(
            result
        )

        stage_start = time.perf_counter()

        audit = await asyncio.wait_for(
            run_output_guardrail(
                llm_response=answer_text,
                retrieved_chunks=guard_context,
                citations=citations,
            ),
            timeout=GUARDRAIL_TIMEOUT,
        )

        logger.info(
            "OUTPUT AUDIT DETAILS: %s",
            audit.model_dump(),
        )

        _log_elapsed(
            "Output guardrail",
            stage_start,
        )

        result.pop(
            "guard_context",
            None,
        )

        result.pop(
            "chunks",
            None,
        )

        logger.info(
            "Total request time: %.2f ms",
            (time.perf_counter() - request_start) * 1000,
        )

        if not audit.passed:
            logger.warning(
                "Output guardrail rejected the generated answer."
            )

            return AskResponse(
                answer=audit.final_answer,
                citations=[],
                effective_date=None,
                is_superseded_warning=False,
            )

        return AskResponse(
            answer=redactor.redact(
                audit.final_answer
            ),
            citations=citations,
            effective_date=result.get(
                "effective_date"
            ),
            is_superseded_warning=result.get(
                "is_superseded_warning",
                False,
            ),
        )

    except asyncio.TimeoutError:
        logger.error(
            "RAG request timed out."
        )

        raise HTTPException(
            status_code=504,
            detail={
                "error_code": "TIMEOUT",
                "message": (
                    "The request exceeded its configured timeout."
                ),
            },
        )

    except HTTPException:
        raise

    except GuardrailTripException:
        raise

    except Exception as exc:
        logger.exception(
            "RAG request failed: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "RAG_FAILURE",
                "message": (
                    "The assistant could not complete the request "
                    "safely. Please retry."
                ),
            },
        )


@app.post(
    "/ask/stream",
    summary="Stream an answer (Server-Sent Events)",
    tags=["RAG"],
)
async def ask_question_stream(
    request: AskRequest,
    current_user: TokenData = Depends(get_current_user),
    redactor: PIIRedactor = Depends(get_redactor),
    llm=Depends(get_llm),
    ranking_pipeline=Depends(get_ranking_pipeline),
):
    safe_query = redactor.redact(
        request.query
    )

    if not llm or not ranking_pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not configured.",
        )

    async def event_generator():
        try:
            input_result = await asyncio.wait_for(
                run_input_guardrail(safe_query),
                timeout=GUARDRAIL_TIMEOUT,
            )

            if not input_result.allowed:
                yield _sse({
                    "type": "final",
                    "data": {
                        "answer": (
                            input_result.refusal_message
                            or "Request blocked by policy."
                        ),
                        "citations": [],
                        "effective_date": None,
                        "is_superseded_warning": False,
                    },
                })
                return

            query = input_result.sanitized_text

            if detect_injection(query):
                raise GuardrailTripException(
                    "Request blocked: prompt-injection pattern detected."
                )

            check_prompt_injection(query)

            pipeline = get_northbridge_pipeline()

            if pipeline is None:
                yield _sse({
                    "type": "error",
                    "code": "RAG_UNAVAILABLE",
                    "message": (
                        "The multimodal RAG pipeline is unavailable."
                    ),
                })
                return

            stage_start = time.perf_counter()

            ctx = await asyncio.wait_for(
                asyncio.to_thread(
                    pipeline._retrieve_combined,
                    query,
                ),
                timeout=RETRIEVAL_TIMEOUT,
            )

            _log_elapsed(
                "Streaming retrieval",
                stage_start,
            )

            ctx = _guard_chunks(ctx)

            ctx = _redact_retrieved_context(
                ctx,
                redactor,
            )

            if not ctx or not getattr(ctx, "docs", None):
                yield _sse({
                    "type": "final",
                    "data": {
                        "answer": (
                            "I cannot answer this question because "
                            "no supporting Northbridge Bank document "
                            "was found."
                        ),
                        "citations": [],
                        "effective_date": None,
                        "is_superseded_warning": False,
                    },
                })
                return

            buffered_tokens = []
            final_payload = None

            async for event in stream_answer(
                question=query,
                ctx=ctx,
                llm=llm,
                timeout_seconds=LLM_TIMEOUT,
            ):
                if event.get("type") == "token":
                    buffered_tokens.append(
                        str(event.get("data", ""))
                    )

                elif event.get("type") == "final":
                    final_payload = event.get(
                        "data",
                        {},
                    )

                elif event.get("type") == "error":
                    yield _sse(event)
                    return

            if final_payload is None:
                yield _sse({
                    "type": "error",
                    "code": "EMPTY_STREAM",
                    "message": (
                        "No validated answer was produced."
                    ),
                })
                return

            answer_text = redactor.redact(
                final_payload.get(
                    "answer",
                    "",
                )
            )

            citations = final_payload.get(
                "citations",
                [],
            )

            _log_citation_debug(
                answer_text,
                citations,
            )

            guard_context = _get_guard_context(
                final_payload
            )

            stage_start = time.perf_counter()

            audit = await asyncio.wait_for(
                run_output_guardrail(
                    llm_response=answer_text,
                    retrieved_chunks=guard_context,
                    citations=citations,
                ),
                timeout=GUARDRAIL_TIMEOUT,
            )

            logger.info(
                "STREAM OUTPUT AUDIT DETAILS: %s",
                audit.model_dump(),
            )

            _log_elapsed(
                "Streaming output guardrail",
                stage_start,
            )

            if not audit.passed:
                yield _sse({
                    "type": "final",
                    "data": {
                        "answer": audit.final_answer,
                        "citations": [],
                        "effective_date": None,
                        "is_superseded_warning": False,
                    },
                })
                return

            for piece in buffered_tokens:
                if piece:
                    yield _sse({
                        "type": "token",
                        "data": redactor.redact(piece),
                    })

            final_payload["answer"] = redactor.redact(
                audit.final_answer
            )

            final_payload.pop(
                "guard_context",
                None,
            )

            final_payload.pop(
                "chunks",
                None,
            )

            yield _sse({
                "type": "final",
                "data": final_payload,
            })

        except asyncio.TimeoutError:
            yield _sse({
                "type": "error",
                "code": "TIMEOUT",
                "message": (
                    "The request timed out. Please try again."
                ),
            })

        except GuardrailTripException as exc:
            yield _sse({
                "type": "error",
                "code": "GUARDRAIL_TRIP",
                "message": str(exc),
            })

        except Exception as exc:
            logger.exception(
                "Streaming RAG request failed: %s",
                type(exc).__name__,
            )

            yield _sse({
                "type": "error",
                "code": "RAG_FAILURE",
                "message": (
                    "The assistant could not complete the request safely."
                ),
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
