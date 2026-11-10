"""
database.py
SQLite persistence layer for the Sign Language Recognition System.
Stores prediction history, emotion readings, generated sentences and
user/session metadata. Uses the standard library sqlite3 module only,
so there is no extra dependency for the database layer.
"""

import sqlite3
import os
import threading
from datetime import datetime

import config

_local = threading.local()


def get_connection():
    """Return a thread-local SQLite connection (SQLite connections are not
    safe to share across threads, which matters because Flask's dev server
    and the camera streaming generator run on different threads)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(config.DATABASE_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db():
    """Create all tables if they do not already exist. Safe to call every
    time the application starts."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            gesture TEXT NOT NULL,
            confidence REAL NOT NULL,
            emotion TEXT,
            emotion_confidence REAL,
            sentence TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_words TEXT NOT NULL,
            corrected_sentence TEXT NOT NULL,
            source TEXT NOT NULL,          -- 'sign' or 'speech'
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dataset_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gesture_label TEXT NOT NULL,
            user_id TEXT NOT NULL,
            sequence_path TEXT NOT NULL,
            num_frames INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    conn.commit()


# ----------------------------------------------------------------------------
# PREDICTIONS
# ----------------------------------------------------------------------------
def insert_prediction(gesture, confidence, emotion=None, emotion_confidence=None, sentence=None):
    conn = get_connection()
    now = datetime.now()
    conn.execute(
        """INSERT INTO predictions
           (date, time, gesture, confidence, emotion, emotion_confidence, sentence, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            gesture,
            float(confidence),
            emotion,
            float(emotion_confidence) if emotion_confidence is not None else None,
            sentence,
            now.isoformat(),
        ),
    )
    conn.commit()


def get_all_predictions(limit=200):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_predictions_by_date(date_str):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE date = ? ORDER BY id DESC", (date_str,)
    ).fetchall()
    return [dict(r) for r in rows]


def clear_predictions():
    conn = get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()


# ----------------------------------------------------------------------------
# SENTENCES
# ----------------------------------------------------------------------------
def insert_sentence(raw_words, corrected_sentence, source="sign"):
    conn = get_connection()
    now = datetime.now()
    conn.execute(
        """INSERT INTO sentences (raw_words, corrected_sentence, source, created_at)
           VALUES (?, ?, ?, ?)""",
        (raw_words, corrected_sentence, source, now.isoformat()),
    )
    conn.commit()


def get_all_sentences(limit=200):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sentences ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------------------------
# DATASET SAMPLES
# ----------------------------------------------------------------------------
def insert_dataset_sample(gesture_label, user_id, sequence_path, num_frames):
    conn = get_connection()
    now = datetime.now()
    conn.execute(
        """INSERT INTO dataset_samples (gesture_label, user_id, sequence_path, num_frames, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (gesture_label, user_id, sequence_path, num_frames, now.isoformat()),
    )
    conn.commit()


def get_dataset_summary():
    """Return {gesture_label: sample_count} for the dashboard."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT gesture_label, COUNT(*) as cnt FROM dataset_samples GROUP BY gesture_label"
    ).fetchall()
    return {r["gesture_label"]: r["cnt"] for r in rows}


# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
def set_setting(key, value):
    conn = get_connection()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    conn.commit()


def get_setting(key, default=None):
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def get_all_settings():
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {config.DATABASE_PATH}")
