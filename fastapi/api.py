#jwt for auth and authorization(optinal-used for session tracing in langfuse)
import jwt

#fastapi imports
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

#session id for tracing in langfuse
import uuid


from models import User, Token, UserResponse
from jwt_utils import (
    verify_password,
    create_access_token,
    decode_access_token,
)
from database import (
    get_user_by_username,
    create_user,
)

# Initialize FastAPI application 
app = FastAPI()

# Configure HTTPBearer security scheme for Swagger UI
# This activates the "Authorize" button in Swagger UI (/docs)
security = HTTPBearer(
    scheme_name="HTTPBearer",
    description="Enter the JWT access token. Swagger UI will automatically prepend 'Bearer ' to the Authorization header.",
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Dependency that extracts the Bearer token from the Authorization header,
    verifies it using jwt_utils, and returns the decoded token payload claims.
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
        return payload

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


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Enforces that the authenticated user has the 'admin' role.
    """
    if current_user.get("role") != "admin":
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
    admin_user: dict = Depends(require_admin),
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




@app.get("/sessionId",
    tags=["Protected"],
)
async def get_users(current_user: dict = Depends(get_current_user)):
    user_session_id = current_user.get("session_id")
    return user_session_id

