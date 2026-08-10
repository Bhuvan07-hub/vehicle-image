"""
Thin SQLite access layer.

We use the stdlib `sqlite3` module directly instead of an ORM (SQLAlchemy /
Tortoise / etc). For a schema this small (two tables) an ORM adds
indirection without much payoff, and staying on stdlib means the persistence
layer has zero third-party dependencies and is trivial to unit test.
A new connection is opened per operation (SQLite connections are cheap) with
WAL mode enabled so concurrent reads don't block the single writer thread.
"""
import sqlite3
import threading
from contextlib import contextmanager

from app.config import DB_PATH

# SQLite allows one writer at a time; a process-wide lock keeps writes from
# a) the request thread (metadata insert) and b) worker threads (status
# updates, result writes) from colliding. Reads don't need the lock.
_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    content_type TEXT,
    file_size_bytes INTEGER,
    width INTEGER,
    height INTEGER,
    sha256_hash TEXT,
    perceptual_hash TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL UNIQUE REFERENCES images(id),
    issues_detected TEXT NOT NULL,
    checks TEXT NOT NULL,
    overall_risk_score REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256_hash);
CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def connection():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_connection():
    """Serializes writers across threads; use for INSERT/UPDATE/DELETE."""
    with _write_lock:
        conn = get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db() -> None:
    with write_connection() as conn:
        conn.executescript(SCHEMA)
