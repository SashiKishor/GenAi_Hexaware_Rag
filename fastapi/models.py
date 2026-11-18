import re
from pydantic import BaseModel, Field, field_validator


class User(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped or len(stripped) < 3 or len(stripped) > 50:
            raise ValueError("Username must be between 3 and 50 characters long.")
        if not re.match(r"^[a-zA-Z0-9_-]+$", stripped):
            raise ValueError("Username can only contain letters, numbers, underscores, or hyphens.")
        return stripped

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        if len(v) < 8 or len(v) > 72:
            raise ValueError("Password must be between 8 and 72 characters long.")
        if " " in v:
            raise ValueError("Password must not contain spaces.")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter (A-Z).")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter (a-z).")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit (0-9).")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>\-_]", v):
            raise ValueError(
                "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>-)."
            )
        return v


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    session_id: str

class TokenData(BaseModel):
    username: str | None = None
    role: str | None = None


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: str | None = None
