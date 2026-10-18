"""Data access for LinguaFlow; SQLAlchemy supports local SQLite and Supabase PostgreSQL."""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column, Date, DateTime, Float, ForeignKey, Integer, MetaData, String, Table,
    Text, UniqueConstraint, create_engine, delete, func, insert, select, update,
)
from sqlalchemy.engine import Engine
from sqlalchemy import inspect, text

LOCAL_DATABASE_URL = f"sqlite:///{Path(__file__).with_name('lingua.db').as_posix()}"
_active_database_url = ""
metadata = MetaData()

users = Table(
    "users", metadata,
    Column("user_id", String(100), primary_key=True),
    Column("email", String(320), nullable=False, unique=True),
    Column("password_hash", String(512)),
    Column("groq_api_key", String(512), nullable=True),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
)
vocabulary = Table(
    "vocabulary", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(100), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=True, index=True),
    Column("term", String(300), nullable=False),
    Column("source_language", String(30), nullable=False),
    Column("vietnamese_meaning", Text, nullable=False),
    Column("other_language", Text, default=""), Column("part_of_speech", String(100), default=""),
    Column("pronunciation", String(300), default=""), Column("examples", Text, default=""),
    Column("synonyms", Text, default=""), Column("antonyms", Text, default=""), Column("tags", String(500), default=""),
    Column("major", String(100), nullable=False, default="Tổng quát"),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
    Column("ease_factor", Float, default=2.5, nullable=False), Column("interval_days", Integer, default=0, nullable=False),
    Column("repetitions", Integer, default=0, nullable=False), Column("due_date", Date, default=date.today, nullable=False),
    UniqueConstraint("user_id", "term", "source_language", "major", name="uq_user_vocab_term_language_major"),
)
study_history = Table(
    "study_history", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(100), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=True, index=True),
    Column("event_type", String(50), nullable=False), Column("details", Text, default=""),
    Column("score", Float), Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
)
quiz_results = Table(
    "quiz_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(100), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=True, index=True),
    Column("quiz_type", String(100), default=""), Column("score", Float, nullable=False),
    Column("total", Integer, nullable=False), Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
)
roadmap_lessons = Table(
    "roadmap_lessons", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(100), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True),
    Column("track", String(20), nullable=False), Column("level", String(30), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
)


@lru_cache(maxsize=4)
def _engine_for(url: str) -> Engine:
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    options: dict[str, Any] = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **options)


def init_db(db_url: str | None = None) -> None:
    """Select DB and create tables. Missing URL uses local lingua.db SQLite."""
    global _active_database_url
    _active_database_url = (db_url or "").strip() or LOCAL_DATABASE_URL
    engine = _engine_for(_active_database_url)
    metadata.create_all(engine)
    # Migrate schemas from earlier LinguaFlow versions in-place (both SQLite and Postgres).
    with engine.begin() as conn:
        user_cols = {column["name"] for column in inspect(conn).get_columns("users")}
        if "groq_api_key" not in user_cols:
            conn.execute(text("ALTER TABLE users ADD COLUMN groq_api_key VARCHAR(512)"))
        for table_name in ("vocabulary", "study_history", "quiz_results"):
            cols = {column["name"] for column in inspect(conn).get_columns(table_name)}
            if "user_id" not in cols:
                conn.execute(text(
                    f"ALTER TABLE {table_name} ADD COLUMN user_id VARCHAR(100) "
                    "REFERENCES users(user_id) ON DELETE CASCADE"
                ))
            if table_name == "vocabulary" and "major" not in cols:
                conn.execute(text("ALTER TABLE vocabulary ADD COLUMN major VARCHAR(100) NOT NULL DEFAULT 'Tổng quát'"))
        if _active_database_url.startswith("sqlite"):
            # Remove the pre-multiuser unique index (term, language) so two
            # separate accounts may independently study the same word.
            conn.execute(text("DROP INDEX IF EXISTS idx_vocab_term_lang"))
            conn.execute(text("DROP INDEX IF EXISTS idx_vocab_user_term_lang"))
        else:
            # Remove a legacy Postgres unique constraint if an older schema had one.
            for constraint in inspect(conn).get_unique_constraints("vocabulary"):
                if set(constraint.get("column_names") or []) in (
                    {"term", "source_language"},
                    {"user_id", "term", "source_language"},
                ) and constraint.get("name"):
                    conn.execute(text(f'ALTER TABLE vocabulary DROP CONSTRAINT "{constraint["name"]}"'))
        conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_vocab_user_term_language_major "
            "ON vocabulary (user_id, term, source_language, major)"
        ))


def _engine() -> Engine:
    if not _active_database_url:
        init_db()
    return _engine_for(_active_database_url)


def _dicts(result) -> list[dict]:
    return [dict(row) for row in result.mappings().all()]


def create_local_user(email: str, password_hash: str, user_id: str) -> bool:
    with _engine().begin() as conn:
        exists = conn.execute(select(users.c.user_id).where(func.lower(users.c.email) == email.lower())).first()
        if exists:
            return False
        conn.execute(insert(users).values(user_id=user_id, email=email.lower(), password_hash=password_hash))
    return True


def get_local_user(email: str) -> dict | None:
    with _engine().connect() as conn:
        row = conn.execute(select(users).where(func.lower(users.c.email) == email.lower())).mappings().first()
        return dict(row) if row else None


def update_user_groq_key(user_id: str, api_key: str | None) -> None:
    """Save a personal Groq key or clear it when an empty value is supplied."""
    normalized_key = (api_key or "").strip() or None
    with _engine().begin() as conn:
        conn.execute(update(users).where(users.c.user_id == user_id).values(groq_api_key=normalized_key))


def get_user_groq_key(user_id: str) -> str | None:
    """Return this user's personal Groq API key, if configured."""
    with _engine().connect() as conn:
        return conn.execute(select(users.c.groq_api_key).where(users.c.user_id == user_id)).scalar_one_or_none()


def ensure_external_user(user_id: str, email: str) -> None:
    with _engine().begin() as conn:
        existing = conn.execute(select(users.c.user_id).where(users.c.user_id == user_id)).first()
        if not existing:
            conn.execute(insert(users).values(user_id=user_id, email=email.lower()))


def claim_legacy_records(user_id: str) -> None:
    """Attach pre-multiuser rows with no owner to the first authenticated account."""
    with _engine().begin() as conn:
        for table in (vocabulary, study_history, quiz_results):
            conn.execute(update(table).where(table.c.user_id.is_(None)).values(user_id=user_id))


def save_vocabulary(item: dict, user_id: str, major: str = "Tổng quát") -> bool:
    keys = ("term", "source_language", "vietnamese_meaning", "other_language", "part_of_speech",
            "pronunciation", "examples", "synonyms", "antonyms", "tags")
    values = {key: str(item.get(key, "")).strip() for key in keys}
    values["major"] = str(item.get("major") or major or "Tổng quát").strip()
    if not all(values[key] for key in ("term", "source_language", "vietnamese_meaning")):
        return False
    values["user_id"] = user_id
    with _engine().begin() as conn:
        duplicate = conn.execute(select(vocabulary.c.id).where(
            vocabulary.c.user_id == user_id, vocabulary.c.term == values["term"],
            vocabulary.c.source_language == values["source_language"],
        )).first()
        if duplicate:
            return False
        conn.execute(insert(vocabulary).values(**values))
    return True


def get_vocabulary(user_id: str, language: str = "Tất cả", tag: str = "", major: str = "Tất cả") -> list[dict]:
    statement = select(vocabulary).where(vocabulary.c.user_id == user_id)
    if language != "Tất cả":
        statement = statement.where(vocabulary.c.source_language == language)
    if tag.strip():
        statement = statement.where(vocabulary.c.tags.ilike(f"%{tag.strip()}%"))
    if major != "Tất cả":
        statement = statement.where(vocabulary.c.major == major)
    with _engine().connect() as conn:
        return _dicts(conn.execute(statement.order_by(vocabulary.c.created_at.desc())))


def get_random_vocabulary(user_id: str, limit: int = 5) -> list[dict]:
    with _engine().connect() as conn:
        return _dicts(conn.execute(select(vocabulary).where(vocabulary.c.user_id == user_id).order_by(func.random()).limit(limit)))


def get_due_cards(user_id: str, limit: int = 20) -> list[dict]:
    with _engine().connect() as conn:
        return _dicts(conn.execute(select(vocabulary).where(
            vocabulary.c.user_id == user_id, vocabulary.c.due_date <= date.today()
        ).order_by(vocabulary.c.due_date, func.random()).limit(limit)))


def update_vocabulary(vocab_id: int, item: dict, user_id: str) -> None:
    editable = {"term", "source_language", "vietnamese_meaning", "other_language", "part_of_speech",
                "pronunciation", "examples", "synonyms", "antonyms", "tags", "major"}
    values = {key: str(value or "").strip() for key, value in item.items() if key in editable}
    if values:
        with _engine().begin() as conn:
            conn.execute(update(vocabulary).where(vocabulary.c.id == vocab_id, vocabulary.c.user_id == user_id).values(**values))


def delete_vocabulary(vocab_id: int, user_id: str) -> None:
    with _engine().begin() as conn:
        conn.execute(delete(vocabulary).where(vocabulary.c.id == vocab_id, vocabulary.c.user_id == user_id))


def review_card(vocab_id: int, quality: int, user_id: str) -> None:
    with _engine().begin() as conn:
        card = conn.execute(select(vocabulary).where(vocabulary.c.id == vocab_id, vocabulary.c.user_id == user_id)).mappings().first()
        if not card:
            return
        ease = max(1.3, card["ease_factor"] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        repetitions = card["repetitions"] + 1 if quality >= 3 else 0
        interval = 1 if quality < 3 or repetitions == 1 else 6 if repetitions == 2 else max(1, round(card["interval_days"] * ease))
        conn.execute(update(vocabulary).where(vocabulary.c.id == vocab_id, vocabulary.c.user_id == user_id).values(
            ease_factor=ease, interval_days=interval, repetitions=repetitions, due_date=date.today() + timedelta(days=interval)))
    log_history(user_id, "flashcard", f"Ôn từ ID {vocab_id}", float(quality))


def log_history(user_id: str, event_type: str, details: str = "", score: float | None = None) -> None:
    with _engine().begin() as conn:
        conn.execute(insert(study_history).values(user_id=user_id, event_type=event_type, details=details, score=score))


def save_quiz_result(user_id: str, quiz_type: str, score: float, total: int) -> None:
    with _engine().begin() as conn:
        conn.execute(insert(quiz_results).values(user_id=user_id, quiz_type=quiz_type, score=score, total=total))
    log_history(user_id, "quiz", quiz_type, score)


def save_roadmap_lesson(user_id: str, track: str, level: str, content: str) -> int:
    with _engine().begin() as conn:
        result = conn.execute(insert(roadmap_lessons).values(user_id=user_id, track=track, level=level, content=content))
        return result.inserted_primary_key[0]


def get_roadmap_lessons(user_id: str, limit: int = 20) -> list[dict]:
    with _engine().connect() as conn:
        return _dicts(conn.execute(select(roadmap_lessons).where(roadmap_lessons.c.user_id == user_id)
                                   .order_by(roadmap_lessons.c.created_at.desc()).limit(limit)))


def get_statistics(user_id: str) -> dict:
    with _engine().connect() as conn:
        word_count = conn.execute(select(func.count()).select_from(vocabulary).where(vocabulary.c.user_id == user_id)).scalar_one()
        timestamps = [row[0] for row in conn.execute(select(study_history.c.created_at).where(
            study_history.c.user_id == user_id).order_by(study_history.c.created_at.desc())).all() if row[0]]
    days = [datetime.fromisoformat(value.replace("Z", "+00:00")).date() if isinstance(value, str) else value.date() for value in timestamps]
    streak, expected = 0, date.today()
    for studied_day in dict.fromkeys(days):
        if studied_day == expected:
            streak += 1
            expected -= timedelta(days=1)
        elif studied_day < expected:
            break
    return {"words": word_count, "streak": streak, "due": len(get_due_cards(user_id, 999))}


def export_csv(user_id: str) -> str:
    rows, output = get_vocabulary(user_id), StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value.isoformat() if hasattr(value, "isoformat") else value for key, value in row.items()})
    return output.getvalue()
