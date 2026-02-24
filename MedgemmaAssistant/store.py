# ============================================================
# SQLite store for saved notes and flashcards.
# ============================================================

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "learning_data.db"


def get_conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn=None):
    close = False
    if conn is None:
        conn = get_conn()
        close = True
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            front TEXT NOT NULL,
            back TEXT NOT NULL,
            topic TEXT,
            next_review TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            conversation_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.commit()
    if close:
        conn.close()


# ----- Notes -----

def save_note(question: str, answer: str, sources: list[str] | None = None) -> int:
    conn = get_conn()
    init_db(conn)
    sources_json = json.dumps(sources) if sources else None
    cur = conn.execute(
        "INSERT INTO notes (question, answer, sources_json, created_at) VALUES (?, ?, ?, ?)",
        (question, answer, sources_json, datetime.utcnow().isoformat()),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_all_notes() -> list[dict]:
    conn = get_conn()
    init_db(conn)
    rows = conn.execute("SELECT id, question, answer, sources_json, created_at FROM notes ORDER BY created_at DESC").fetchall()
    conn.close()
    out = []
    for r in rows:
        sources = json.loads(r["sources_json"]) if r["sources_json"] else None
        out.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "sources": sources,
            "created_at": r["created_at"],
        })
    return out


# ----- Flashcards -----

def save_flashcard(front: str, back: str, topic: str | None = None) -> int:
    conn = get_conn()
    init_db(conn)
    cur = conn.execute(
        "INSERT INTO flashcards (front, back, topic, next_review, created_at) VALUES (?, ?, ?, ?, ?)",
        (front, back, topic or "", datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def save_flashcards(cards: list[dict], topic: str | None = None) -> int:
    """cards = [{"front": "...", "back": "..."}, ...]"""
    conn = get_conn()
    init_db(conn)
    now = datetime.utcnow().isoformat()
    for c in cards:
        conn.execute(
            "INSERT INTO flashcards (front, back, topic, next_review, created_at) VALUES (?, ?, ?, ?, ?)",
            (c.get("front", ""), c.get("back", ""), topic or "", now, now),
        )
    conn.commit()
    count = len(cards)
    conn.close()
    return count


def get_all_flashcards(topic: str | None = None) -> list[dict]:
    conn = get_conn()
    init_db(conn)
    if topic:
        rows = conn.execute(
            "SELECT id, front, back, topic, next_review, created_at FROM flashcards WHERE topic = ? OR topic = '' ORDER BY id",
            (topic,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, front, back, topic, next_review, created_at FROM flashcards ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_flashcards_for_review(limit: int = 20) -> list[dict]:
    """Return cards due for review (next_review <= now or null)."""
    conn = get_conn()
    init_db(conn)
    now = datetime.utcnow().isoformat()
    rows = conn.execute(
        "SELECT id, front, back, topic FROM flashcards WHERE next_review IS NULL OR next_review <= ? ORDER BY id LIMIT ?",
        (now, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_flashcard_review(card_id: int, next_review_iso: str) -> None:
    conn = get_conn()
    conn.execute("UPDATE flashcards SET next_review = ? WHERE id = ?", (next_review_iso, card_id))
    conn.commit()
    conn.close()


# ----- Conversations -----

def save_conversation(conversation: list[dict], title: str | None = None) -> int:
    """Save a whole conversation to the database."""
    conn = get_conn()
    init_db(conn)
    
    # Generate title if not provided (use first question)
    if not title and conversation:
        for item in conversation:
            if item.get("question"):
                title = item["question"][:50] + "..." if len(item["question"]) > 50 else item["question"]
                break
    if not title:
        title = "Untitled Conversation"
    
    now = datetime.utcnow().isoformat()
    conversation_json = json.dumps(conversation)
    
    cur = conn.execute(
        "INSERT INTO conversations (title, conversation_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (title, conversation_json, now, now),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_all_conversations() -> list[dict]:
    """Retrieve all saved conversations."""
    conn = get_conn()
    init_db(conn)
    rows = conn.execute(
        "SELECT id, title, conversation_json, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "title": r["title"],
            "conversation": json.loads(r["conversation_json"]),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    return out


def get_conversation(conversation_id: int) -> dict | None:
    """Get a specific conversation by ID."""
    conn = get_conn()
    init_db(conn)
    row = conn.execute(
        "SELECT id, title, conversation_json, created_at, updated_at FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    conn.close()
    
    if row:
        return {
            "id": row["id"],
            "title": row["title"],
            "conversation": json.loads(row["conversation_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    return None


def delete_conversation(conversation_id: int) -> bool:
    """Delete a conversation by ID."""
    conn = get_conn()
    init_db(conn)
    cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def update_conversation(conversation_id: int, conversation: list[dict], title: str | None = None) -> bool:
    """Update an existing conversation."""
    conn = get_conn()
    init_db(conn)
    
    now = datetime.utcnow().isoformat()
    conversation_json = json.dumps(conversation)
    
    if title:
        cur = conn.execute(
            "UPDATE conversations SET title = ?, conversation_json = ?, updated_at = ? WHERE id = ?",
            (title, conversation_json, now, conversation_id),
        )
    else:
        cur = conn.execute(
            "UPDATE conversations SET conversation_json = ?, updated_at = ? WHERE id = ?",
            (conversation_json, now, conversation_id),
        )
    
    updated = cur.rowcount > 0
    conn.commit()
    conn.close()
    return updated
