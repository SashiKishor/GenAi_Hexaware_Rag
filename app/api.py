#jwt for auth and authorization(optinal-used for session tracing in langfuse)
import jwt

#fastapi imports
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
#session id for tracing in langfuse
import uuid

import os
from dotenv import load_dotenv
load_dotenv()
import logging
import sys
from pathlib import Path
import asyncio
from functools import lru_cache

#Custom Exception
from app.custom_exception import GuardrailTripException

from security.Pii_Redactor import PIIRedactor, pii_redactor
from security.input_guard import run_input_guardrail
from security.output_guard import run_output_guardrail
from security.llam import check_prompt_injection, filter_safe_chunks
from rag.chain import generate_answer, stream_answer
from rag.reranking import detect_injection
from langchain_core.documents import Document

from app.models import User, Token, TokenData, AskRequest, AskResponse

from app.jwt_utils import (
    verify_password,
    create_access_token,
    decode_access_token,
)
from app.database import (
    get_user_by_username,
    create_user,
)

logger = logging.getLogger(__name__)

# Initialize FastAPI application 
app = FastAPI()

# 1. Handles guardrail blocks (NeMo / Llama Guard)
@app.exception_handler(GuardrailTripException)
async def guardrail_handler(request: Request, exc: GuardrailTripException):
    return JSONResponse(
        status_code=400,
        content={"success": False, "error_code": "GUARDRAIL_TRIP", "message": str(exc)}
    )

# 2. Handles LLM / Chroma timeouts
@app.exception_handler(asyncio.TimeoutError)
async def timeout_handler(request: Request, exc: asyncio.TimeoutError):
    return JSONResponse(
        status_code=504,
        content={"success": False, "error_code": "TIMEOUT", "message": "Request timed out. Please try again."}
    )

# 3. Catch-all safety net (prevents any raw 500 crash from leaking)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error_code": "INTERNAL_ERROR", "message": "Something went wrong. Please contact support."}
    )

# Cached singleton dependency: lazily initialized on first request
@lru_cache()
def get_redactor() -> PIIRedactor:
    """
    Cached singleton dependency for PII Redactor.
    Initializes lazily only on the first request that calls for it.
    """
    return pii_redactor


@lru_cache()
def get_llm():
    """
    Returns an initialized LangChain ChatModel (Gemini or OpenAI).
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
        return ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, temperature=0)


    return None


@lru_cache()
def get_ranking_pipeline():
    """Build the hybrid retrieval + version-aware ranking pipeline once."""
    try:
        from vector_store.chroma_db import ChromaDB
        from rag.retrieval import HybridRetriever
        from rag.ranking_pipeline import build_ranking_pipeline, RankingConfig
        db = ChromaDB()
        retriever = HybridRetriever(db.get_vector_store())
        floor = os.getenv("MIN_RELEVANCE", "0.15")
        config = RankingConfig(min_relevance=float(floor) if floor else None)
        return build_ranking_pipeline(retriever, config=config)
    except Exception as exc:
        logger.warning("Ranking pipeline could not be initialized: %s", type(exc).__name__)
        return None


@lru_cache()
def get_northbridge_pipeline():
    """Connect text ranking and multimodal retrieval behind one API pipeline."""
    ranking = get_ranking_pipeline()
    llm = get_llm()
    if ranking is None or llm is None:
        return None
    try:
        from rag.full_pipeline import NorthbridgeRAGPipeline
        return NorthbridgeRAGPipeline(ranking, llm)
    except Exception as exc:
        logger.warning("Multimodal pipeline could not be initialized: %s", type(exc).__name__)
        return None


def _redact_retrieved_context(ctx, redactor: PIIRedactor):
    """Redact retrieved passages before they enter any model prompt or trace."""
    if ctx is None or not hasattr(ctx, "docs"):
        return ctx
    ctx.docs = [Document(page_content=redactor.redact(doc.page_content), metadata=dict(doc.metadata))
                for doc in ctx.docs]
    ctx.superseded_support = [
        Document(page_content=redactor.redact(doc.page_content), metadata=dict(doc.metadata))
        for doc in getattr(ctx, "superseded_support", [])
    ]
    return ctx


def _guard_chunks(ctx):
    if ctx is not None and hasattr(ctx, "docs"):
        # Deterministic injection filter also covers visual chunks added after text reranking.
        ctx.docs = [d for d in ctx.docs if not detect_injection(d.page_content)]
        ctx.superseded_support = [
            d for d in getattr(ctx, "superseded_support", [])
            if not detect_injection(d.page_content)
        ]
        # Optional model-based Llama Guard adds a second independent screen.
        ctx.docs = filter_safe_chunks(ctx.docs)
        ctx.superseded_support = filter_safe_chunks(ctx.superseded_support)
    return ctx


def _sse(event: dict) -> str:
    import json
    return "data: " + json.dumps(event, ensure_ascii=False, default=str) + "\n\n"

# Configure HTTPBearer security scheme for Swagger UI
# This activates the "Authorize" button in Swagger UI (/docs)
security = HTTPBearer(
    scheme_name="HTTPBearer",
    description="Enter the JWT access token. Swagger UI will automatically prepend 'Bearer ' to the Authorization header.",
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """
    Dependency that extracts the Bearer token from the Authorization header,
    verifies it using jwt_utils, and returns a validated TokenData object.
    """
    token = credentials.credentials
    try:
        payload = decode_access_token(token)
        username: str | None = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload is missing user subject ('sub').",
                headers={"WWW-Authenticate": "Bearer"},
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
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token or credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_admin(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """
    Enforces that the authenticated user has the 'admin' role.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Admin role required.",
        )
    return current_user


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post(
    "/login",
    response_model=Token,
    summary="User Login",
    tags=["Authentication"],
)
async def login(user_credentials: User):
    # Query user record from SQLite database
    db_user = get_user_by_username(user_credentials.username)
    if not db_user or not verify_password(user_credentials.password, db_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    session_id = str(uuid.uuid4())
    # Generate JWT access token with claims: 'sub' (username) and 'role'
    access_token = create_access_token(
        data={"sub": db_user["username"], "role": db_user.get("role", "user"),"session_id": session_id}
    )
    return Token(access_token=access_token, token_type="bearer", session_id=session_id)


@app.post(
    "/register",
    summary="Register New User",
    tags=["Authentication"],
    status_code=status.HTTP_201_CREATED,
)
async def register(user_credentials: User):
    existing_user = get_user_by_username(user_credentials.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User '{user_credentials.username}' is already registered in SQLite database.",
        )

    created = create_user(
        username=user_credentials.username,
        plain_password=user_credentials.password,
        role="user",
    )
    return {
        "message": f"User '{user_credentials.username}' registered successfully."
    }


@app.post(
    "/registerManager",
    summary="Register Manager (Admin Only)",
    tags=["Authentication"],
    status_code=status.HTTP_201_CREATED,
)
async def register_manager(
    user_credentials: User,
    admin_user: TokenData = Depends(require_admin),
):
    existing_user = get_user_by_username(user_credentials.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User '{user_credentials.username}' is already registered in SQLite database.",
        )

    created = create_user(
        username=user_credentials.username,
        plain_password=user_credentials.password,
        role="manager",
    )
    return {
        "message": f"Manager '{user_credentials.username}' registered successfully."
    }


@app.get(
    "/sessionId",
    tags=["Protected"],
)
async def get_session_id(current_user: TokenData = Depends(get_current_user)):
    return {"session_id": current_user.session_id, "username": current_user.username, "role": current_user.role}


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
    safe_query = redactor.redact(request.query)
    if not llm or not ranking_pipeline:
        raise HTTPException(status_code=503, detail={
            "error_code": "RAG_UNAVAILABLE",
            "message": "The retrieval or language model service is not configured.",
        })

    # Input classifier, advice refusal, scope check and PII sanitization.
    input_result = await run_input_guardrail(safe_query)
    if not input_result.allowed:
        return AskResponse(answer=input_result.refusal_message or "Request blocked by policy.",
                           citations=[], effective_date=None, is_superseded_warning=False)
    safe_query = input_result.sanitized_text
    if detect_injection(safe_query):
        raise GuardrailTripException("Request blocked: prompt-injection pattern detected.")
    check_prompt_injection(safe_query)

    try:
        pipeline = get_northbridge_pipeline()
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Multimodal RAG pipeline unavailable.")
        ctx = await asyncio.wait_for(asyncio.to_thread(pipeline._retrieve_combined, safe_query), timeout=30)
        ctx = _redact_retrieved_context(_guard_chunks(ctx), redactor)
        result = await generate_answer(question=safe_query, ctx=ctx, llm=llm, timeout_seconds=30)

        output_chunks = [
            {"document_id": c.metadata.get("source", "unknown"),
             "version": c.metadata.get("version", "unknown"),
             "effective_date": c.metadata.get("effective_date", "unknown"),
             "status": c.metadata.get("status", c.metadata.get("version_status", "current")),
             "page": c.metadata.get("page", "N/A"), "content": c.page_content}
            for c in (list(ctx.docs) + list(getattr(ctx, "superseded_support", [])))
        ]
        audit = await run_output_guardrail(result["answer"], output_chunks)
        if not audit.passed:
            return AskResponse(answer=audit.final_answer, citations=[],
                               effective_date=None, is_superseded_warning=False)
        return AskResponse(
            answer=redactor.redact(audit.final_answer),
            citations=result.get("citations", []),
            effective_date=result.get("effective_date"),
            is_superseded_warning=result.get("is_superseded_warning", False),
        )
    except asyncio.TimeoutError:
        raise
    except HTTPException:
        raise
    except GuardrailTripException:
        raise
    except Exception as exc:
        logger.error("RAG request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail={
            "error_code": "RAG_FAILURE",
            "message": "The assistant could not complete the request safely. Please retry.",
        })


@app.post("/ask/stream", summary="Stream an answer (Server-Sent Events)", tags=["RAG"])
async def ask_question_stream(
    request: AskRequest,
    current_user: TokenData = Depends(get_current_user),
    redactor: PIIRedactor = Depends(get_redactor),
    llm=Depends(get_llm),
    ranking_pipeline=Depends(get_ranking_pipeline),
):
    """Authenticated SSE endpoint; retrieved PII is redacted before prompt construction."""
    safe_query = redactor.redact(request.query)
    if not llm or not ranking_pipeline:
        raise HTTPException(status_code=503, detail="RAG service is not configured.")

    async def event_generator():
        try:
            input_result = await run_input_guardrail(safe_query)
            if not input_result.allowed:
                yield _sse({"type": "final", "data": {
                    "answer": input_result.refusal_message, "citations": [],
                    "effective_date": None, "is_superseded_warning": False}})
                return
            query = input_result.sanitized_text
            if detect_injection(query):
                raise GuardrailTripException("Request blocked: prompt-injection pattern detected.")
            check_prompt_injection(query)
            pipeline = get_northbridge_pipeline()
            if pipeline is None:
                yield _sse({"type": "error", "code": "RAG_UNAVAILABLE",
                            "message": "The multimodal RAG pipeline is unavailable."})
                return
            ctx = await asyncio.wait_for(asyncio.to_thread(pipeline._retrieve_combined, query), timeout=30)
            ctx = _redact_retrieved_context(_guard_chunks(ctx), redactor)
            buffered_tokens = []
            final_payload = None
            async for event in stream_answer(question=query, ctx=ctx, llm=llm, timeout_seconds=30):
                if event.get("type") == "token":
                    buffered_tokens.append(str(event.get("data", "")))
                elif event.get("type") == "final":
                    final_payload = event.get("data", {})
                elif event.get("type") == "error":
                    yield _sse(event)
                    return

            if final_payload is None:
                yield _sse({"type": "error", "code": "EMPTY_STREAM",
                            "message": "No validated answer was produced."})
                return

            # Do not release generated tokens until the complete answer passes output checks.
            answer_text = redactor.redact(final_payload.get("answer", ""))
            output_chunks = [
                {"document_id": c.metadata.get("source", "unknown"),
                 "version": c.metadata.get("version", "unknown"),
                 "effective_date": c.metadata.get("effective_date", "unknown"),
                 "status": c.metadata.get("status", c.metadata.get("version_status", "current")),
                 "page": c.metadata.get("page", "N/A"), "content": c.page_content}
                for c in (list(ctx.docs) + list(getattr(ctx, "superseded_support", [])))
            ]
            audit = await run_output_guardrail(answer_text, output_chunks)
            if not audit.passed:
                yield _sse({"type": "final", "data": {
                    "answer": audit.final_answer, "citations": [],
                    "effective_date": None, "is_superseded_warning": False}})
                return
            # Emit only after audit; token chunks preserve the SSE client contract.
            for piece in buffered_tokens:
                if piece:
                    yield _sse({"type": "token", "data": redactor.redact(piece)})
            final_payload["answer"] = redactor.redact(audit.final_answer)
            yield _sse({"type": "final", "data": final_payload})
        except asyncio.TimeoutError:
            yield _sse({"type": "error", "code": "TIMEOUT", "message": "The request timed out. Please try again."})
        except GuardrailTripException as exc:
            yield _sse({"type": "error", "code": "GUARDRAIL_TRIP", "message": str(exc)})
        except Exception as exc:
            logger.error("Streaming RAG request failed: %s", type(exc).__name__)
            yield _sse({"type": "error", "code": "RAG_FAILURE",
                        "message": "The assistant could not complete the request safely."})

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
