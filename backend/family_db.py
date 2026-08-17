"""
Family Circles -- groups of verified contacts with a head-of-family
approval step before a member's face becomes trusted for matching.

Auth model is deliberately simple for this prototype: a shared passcode per
family (set when the family is created), not real per-user accounts. Anyone
with the family_id + passcode can request to join or act as the head to
approve/reject requests. This is fine for a demo/single-household use case;
it is NOT meant to be treated as production-grade access control.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent / "family.db"


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
            CREATE TABLE IF NOT EXISTS families (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                passcode TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS family_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                family_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                face_embedding TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                approved_at TEXT,
                FOREIGN KEY (family_id) REFERENCES families (id)
            )
            """
        )
        # Older DBs created before voice support was removed still have a
        # voice_embedding column on disk -- harmless, just unused: SQLite
        # doesn't require every column to appear in every INSERT/SELECT, and
        # CREATE TABLE IF NOT EXISTS above won't touch an existing table.


def create_family(name: str, passcode: str) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO families (name, passcode) VALUES (?, ?)", (name, passcode)
        )
        return cursor.lastrowid


def verify_family_passcode(family_id: int, passcode: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT passcode FROM families WHERE id = ?", (family_id,)
        ).fetchone()
        return row is not None and row[0] == passcode


def create_member_request(
    family_id: int,
    name: str,
    face_embedding: np.ndarray | None,
) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO family_members (family_id, name, status, face_embedding) "
            "VALUES (?, ?, 'pending', ?)",
            (
                family_id,
                name,
                json.dumps(face_embedding.tolist()) if face_embedding is not None else None,
            ),
        )
        return cursor.lastrowid


def list_members(family_id: int) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, status, created_at, approved_at, "
            "face_embedding IS NOT NULL AS has_face "
            "FROM family_members WHERE family_id = ? ORDER BY id DESC",
            (family_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def set_member_status(member_id: int, status: str) -> None:
    with _connect() as conn:
        if status == "approved":
            conn.execute(
                "UPDATE family_members SET status = ?, approved_at = datetime('now') WHERE id = ?",
                (status, member_id),
            )
        else:
            conn.execute(
                "UPDATE family_members SET status = ? WHERE id = ?", (status, member_id)
            )


def get_approved_member_embeddings(member_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT name, face_embedding FROM family_members "
            "WHERE id = ? AND status = 'approved'",
            (member_id,),
        ).fetchone()
        if not row:
            return None
        name, face_json = row
        return {
            "name": name,
            "face_embedding": np.array(json.loads(face_json), dtype=np.float32) if face_json else None,
        }
