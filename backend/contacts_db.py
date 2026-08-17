"""
Local storage of enrolled contacts' reference face embeddings, for identity
verification (face_embedding.py). Same SQLite-local-file pattern as
history_db.py.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent / "contacts.db"


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
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )


def insert_contact(name: str, embedding: np.ndarray) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO contacts (name, embedding) VALUES (?, ?)",
            (name, json.dumps(embedding.tolist())),
        )
        return cursor.lastrowid


def get_contact_embedding(contact_id: int) -> tuple[str, np.ndarray] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT name, embedding FROM contacts WHERE id = ?", (contact_id,)
        ).fetchone()
        if not row:
            return None
        name, embedding_json = row
        return name, np.array(json.loads(embedding_json), dtype=np.float32)


def list_contacts() -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, created_at FROM contacts ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
