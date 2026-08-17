"""
Lightweight local history of predictions, for the dashboard page.
SQLite, no server, no new heavy dependency -- fine for a single-user local
tool like this.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "history.db"


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                platform TEXT,
                fake_probability REAL NOT NULL,
                thumbnail_b64 TEXT NOT NULL
            )
            """
        )
        # Added for identity verification + scam score -- guarded so this
        # stays idempotent against an already-existing older-schema table.
        for column_def in (
            "identity_contact_name TEXT",
            "identity_similarity REAL",
            "scam_likelihood REAL",
        ):
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {column_def}")
            except sqlite3.OperationalError:
                pass  # column already exists


def insert_prediction(
    platform: str | None,
    fake_probability: float,
    thumbnail_b64: str,
    identity_contact_name: str | None = None,
    identity_similarity: float | None = None,
    scam_likelihood: float | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO predictions "
            "(platform, fake_probability, thumbnail_b64, identity_contact_name, identity_similarity, scam_likelihood) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                platform,
                fake_probability,
                thumbnail_b64,
                identity_contact_name,
                identity_similarity,
                scam_likelihood,
            ),
        )


def get_recent(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, platform, fake_probability, thumbnail_b64, "
            "identity_contact_name, identity_similarity, scam_likelihood "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
