#jwt for auth and authorization(optinal-used for session tracing in langfuse)
import jwt

#fastapi imports
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
#session id for tracing in langfuse
import uuid

import os
import logging
import sys
from pathlib import Path
import asyncio
from functools import lru_cache

#Custom Exception
from app.custom_exception import GuardrailTripException

from security.Pii_Redactor import PIIRedactor
from rag.chain import generate_answer

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
async def guardrail_handler(request: Request, exc: GuardrailTripException):
    return JSONResponse(
        status_code=200,
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
    return PIIRedactor()


@lru_cache()
def get_llm():
    """
    Returns an initialized LangChain ChatModel (Gemini or OpenAI).
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, temperature=0)


    return None


@lru_cache()
def get_ranking_pipeline():
    """
    Returns the RankingPipeline using ChromaDB and Retriever if available.
    """
    try:
        from vector_store.chroma_db import ChromaDB
        from vector_store.retriever import Retriever
        from rag.ranking_pipeline import build_ranking_pipeline

        db = ChromaDB()
        retriever = Retriever(db)
        return build_ranking_pipeline(retriever)
    except Exception as e:
        logger.warning("Ranking pipeline could not be initialized: %s", e)
        return None

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
    # 1. Redact PII before any logging or LLM call (F6)
    safe_query = redactor.redact(request.query)

    # 2. Retrieve & rank documents using the ranking pipeline (if available)
    ctx = None
    if ranking_pipeline:
        try:
            ctx = ranking_pipeline.ask(safe_query)
        except Exception as e:
            logger.error("Ranking pipeline search failed: %s", e)
            ctx = None

    # 3. If documents were found but LLM is not configured, fail safely
    if ctx and hasattr(ctx, "answerable") and ctx.answerable and not llm:
        return AskResponse(
            answer="I cannot answer this right now because the LLM service is not configured.",
            citations=[],
            effective_date=None,
            is_superseded_warning=False,
        )

    # 4. Generate answer via rag.chain.generate_answer
    result = await generate_answer(
        question=safe_query,
        ctx=ctx,
        llm=llm,
        timeout_seconds=30,
    )

    # 5. Return the validated response (F1, F3, F5)
    return AskResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        effective_date=result.get("effective_date"),
        is_superseded_warning=result.get("is_superseded_warning", False),
    )

