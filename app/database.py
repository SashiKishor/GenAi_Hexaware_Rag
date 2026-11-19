import re
import sqlite3
from pathlib import Path
from typing import Optional

from app.jwt_utils import hash_password

# Database file location
DB_PATH = Path(__file__).resolve().parent / "users.db"
USER_TXT_PATH = Path(__file__).resolve().parent / "user.txt"

# Default seed credentials if user.txt is missing
DEFAULT_SEED_USERS = [
    ("john_doe", "SecurePass123!", "admin"),
    ("alice_smith", "AlicePass2026@", "user"),
    ("bob_wilson", "BobPass2026#", "user"),
    ("carol_danvers", "CarolPass2026$", "manager"),
]


def get_connection() -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Sets row_factory to sqlite3.Row for dict-like column access.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initializes the SQLite database:
    1. Creates the 'users' table if it does not already exist.
    2. Seeds the table with initial users from user.txt if empty.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        # Check if users table is empty
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        if count == 0:
            seed_users_from_file_or_defaults(conn)


def parse_users_from_txt() -> list[tuple[str, str, str]]:
    """
    Parses usernames, passwords, and roles from user.txt.
    Lines matching format: 'username | password | role' are extracted.
    """
    users = []
    if not USER_TXT_PATH.exists():
        return DEFAULT_SEED_USERS

    with open(USER_TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignore comments, dividers, and empty lines
            if not line or line.startswith("#") or line.startswith("=") or line.startswith("-") or line.startswith("FORMAT"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                username = parts[0]
                password = parts[1]
                role = parts[2] if len(parts) >= 3 else "user"
                users.append((username, password, role))

    return users if users else DEFAULT_SEED_USERS


def seed_users_from_file_or_defaults(conn: sqlite3.Connection) -> None:
    """
    Seeds initial users into the SQLite database with bcrypt-hashed passwords.
    """
    users_to_seed = parse_users_from_txt()
    cursor = conn.cursor()

    for username, plain_password, role in users_to_seed:
        hashed = hash_password(plain_password)
        cursor.execute(
            """
            INSERT OR IGNORE INTO users (username, hashed_password, role)
            VALUES (?, ?, ?)
            """,
            (username, hashed, role),
        )
    conn.commit()


def get_user_by_username(username: str) -> Optional[dict]:
    """
    Fetches a single user record from SQLite by username.
    Returns a dict containing id, username, hashed_password, role, created_at or None.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, username, hashed_password, role, created_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def create_user(username: str, plain_password: str, role: str = "user") -> dict:
    """
    Hashes the password with bcrypt and inserts a new user record into SQLite.
    Also appends the credentials to user.txt.
    """
    hashed = hash_password(plain_password)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (username, hashed_password, role)
            VALUES (?, ?, ?)
            """,
            (username, hashed, role),
        )
        conn.commit()
        user_id = cursor.lastrowid


    return {
        "username": username,
        "role": role,
    }


def append_user_to_txt(username: str, plain_password: str, role: str = "user") -> None:
    """
    Appends newly registered user credentials to user.txt.
    """
    try:
        with open(USER_TXT_PATH, "a", encoding="utf-8") as f:
            f.write(f"{username} | {plain_password} | {role}\n")
    except Exception as e:
        print(f"Warning: Could not append user to {USER_TXT_PATH}: {e}")



def ensure_db_exists() -> None:
    """
    Checks if SQLite database exists and is populated.
    Initializes and seeds only if the database file is missing or empty.
    """
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        init_db()


# Automatically initialize only when there is no database file
ensure_db_exists()

