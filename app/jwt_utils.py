import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import bcrypt
import jwt
from dotenv import load_dotenv

# Locate and load the .env file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# JWT Configuration strictly loaded from environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
expire_minutes_str = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set. Please define it in the .env file.")
if not ALGORITHM:
    raise RuntimeError("ALGORITHM is not set. Please define it in the .env file.")
if not expire_minutes_str:
    raise RuntimeError("ACCESS_TOKEN_EXPIRE_MINUTES is not set. Please define it in the .env file.")

ACCESS_TOKEN_EXPIRE_MINUTES = int(expire_minutes_str)


def hash_password(password: str) -> str:
    """
    Hashes a plaintext password using bcrypt with a securely generated salt.
    Args:
        password (str): Plaintext password to hash.
    Returns:
        str: Bcrypt-hashed password decoded as a UTF-8 string.
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plaintext password against a stored bcrypt hash.
    Args:
        plain_password (str): The plaintext password provided by the user.
        hashed_password (str): The bcrypt-hashed password string.
    Returns:
        bool: True if password matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Generates an encoded JWT access token with expiration and issue-time claims.
    Args:
        data (dict): The payload/claims to encode into the token (e.g., {"sub": username}).
        expires_delta (timedelta | None): Optional custom lifespan for the token.
    Returns:
        str: Encoded JWT string signed with the secret key and algorithm.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": now})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """
    Decodes and validates a JWT access token.
    Args:
        token (str): The JWT string to decode.
    Returns:
        dict: The decoded token claims/payload.
    Raises:
        jwt.ExpiredSignatureError: If the token has expired.
        jwt.InvalidTokenError: If the token is invalid or signature verification fails.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
