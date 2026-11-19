#jwt for auth and authorization(optinal-used for session tracing in langfuse)
import jwt

#fastapi imports
from fastapi import FastAPI, Depends, HTTPException, status,Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
#session id for tracing in langfuse
import uuid

import sys
from pathlib import Path
import asyncio

#Custom Exception
from app.custom_exception import GuardrailTripException

from security.Pii_Redactor import PIIRedactor

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

from functools import lru_cache

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
):
    # 1. Redact PII before any logging or LLM call (F6)
    safe_query = redactor.redact(request.query)

    if "invest" in safe_query.lower():
        raise GuardrailTripException("I cannot give investment advice. Please refer to a licensed advisor.")
    # 2. Call your RAG / LangGraph pipeline (replace with your actual function when built)
    # Example:
    # result = await run_rag_pipeline(
    #     query=safe_query,
    #     session_id=current_user.session_id,
    #     role=current_user.role
    # )
    result = None

    # 3. Safe fallback if no documents matched or pipeline is offline (F9)
    if not result or not result.get("citations"):
        return AskResponse(
            answer="I cannot answer this based on Northbridge Bank's current verified documentation.",
            citations=[],
            effective_date=None,
            is_superseded_warning=False,
        )

    # 4. Return the validated response (F1, F3, F5)
    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        effective_date=result.get("effective_date"),
        is_superseded_warning=result.get("is_superseded_warning", False),
    )

