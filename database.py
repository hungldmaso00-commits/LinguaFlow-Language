"""Database layer for LinguaFlow.

Uses PostgreSQL/Supabase when DATABASE_URL is supplied; otherwise creates the
local lingua.db SQLite file so the same app works locally and on cloud.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any

from sqlalchemy import Column, Date, DateTime, Float, Integer, MetaData, String, Table, Text, UniqueConstraint, create_engine, delete, func, insert, select, update
from sqlalchemy.engine import Engine

LOCAL_DATABASE_URL = f"sqlite:///{Path(__file__).with_name('lingua.db').as_posix()}"
_active_database_url = ""
metadata = MetaData()

vocabulary = Table(
    "vocabulary", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("term", String(300), nullable=False),
    Column("source_language", String(30), nullable=False),
    Column("vietnamese_meaning", Text, nullable=False),
    Column("other_language", Text, default=""),
    Column("part_of_speech", String(100), default=""),
    Column("pronunciation", String(300), default=""),
    Column("examples", Text, default=""),
    Column("synonyms", Text, default=""),
    Column("antonyms", Text, default=""),
    Column("tags", String(500), default=""),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
    Column("ease_factor", Float, default=2.5, nullable=False),
    Column("interval_days", Integer, default=0, nullable=False),
    Column("repetitions", Integer, default=0, nullable=False),
    Column("due_date", Date, default=date.today, nullable=False),
    UniqueConstraint("term", "source_language", name="uq_vocabulary_term_language"),
)
study_history = Table(
    "study_history", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_type", String(50), nullable=False),
    Column("details", Text, default=""),
    Column("score", Float, nullable=True),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
)
quiz_results = Table(
    "quiz_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("quiz_type", String(100), default=""),
    Column("score", Float, nullable=False),
    Column("total", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
)


@lru_cache(maxsize=4)
def _create_engine(db_url: str) -> Engine:
    """Create and cache an SQLAlchemy engine for a URL."""
    options: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if db_url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(db_url, **options)


def init_db(db_url: str | None = None) -> None:
    """Select the active database and create tables if they do not yet exist.

    db_url is a PostgreSQL/Supabase URL. When omitted or empty, SQLite is used.
    """
    global _active_database_url
    _active_database_url = (db_url or "").strip() or LOCAL_DATABASE_URL
    metadata.create_all(_create_engine(_active_database_url))


def _engine() -> Engine:
    """Return the initialized engine; initialize local SQLite for direct scripts."""
    if not _active_database_url:
        init_db()
    return _create_engine(_active_database_url)


def _as_dicts(result) -> list[dict]:
    return [dict(row) for row in result.mappings().all()]


def save_vocabulary(item: dict) -> bool:
    """Insert an item, returning False when it is invalid or already exists."""
    fields = ("term", "source_language", "vietnamese_meaning", "other_language",
              "part_of_speech", "pronunciation", "examples", "synonyms",
              "antonyms", "tags")
    values = {field: str(item.get(field, "")).strip() for field in fields}
    if not all(values[key] for key in ("term", "source_language", "vietnamese_meaning")):
        return False
    with _engine().begin() as connection:
        duplicate = connection.execute(select(vocabulary.c.id).where(
            vocabulary.c.term == values["term"],
            vocabulary.c.source_language == values["source_language"],
        )).first()
        if duplicate:
            return False
        connection.execute(insert(vocabulary).values(**values))
    return True


def get_vocabulary(language: str = "Tất cả", tag: str = "") -> list[dict]:
    statement = select(vocabulary)
    if language != "Tất cả":
        statement = statement.where(vocabulary.c.source_language == language)
    if tag.strip():
        statement = statement.where(vocabulary.c.tags.ilike(f"%{tag.strip()}%"))
    with _engine().connect() as connection:
        return _as_dicts(connection.execute(statement.order_by(vocabulary.c.created_at.desc())))


def get_random_vocabulary(limit: int = 5) -> list[dict]:
    with _engine().connect() as connection:
        return _as_dicts(connection.execute(select(vocabulary).order_by(func.random()).limit(limit)))


def get_due_cards(limit: int = 20) -> list[dict]:
    with _engine().connect() as connection:
        return _as_dicts(connection.execute(
            select(vocabulary).where(vocabulary.c.due_date <= date.today())
            .order_by(vocabulary.c.due_date, func.random()).limit(limit)
        ))


def update_vocabulary(vocab_id: int, item: dict) -> None:
    editable = {"term", "source_language", "vietnamese_meaning", "other_language",
                "part_of_speech", "pronunciation", "examples", "synonyms",
                "antonyms", "tags"}
    values = {key: str(value or "").strip() for key, value in item.items() if key in editable}
    if values:
        with _engine().begin() as connection:
            connection.execute(update(vocabulary).where(vocabulary.c.id == vocab_id).values(**values))


def delete_vocabulary(vocab_id: int) -> None:
    with _engine().begin() as connection:
        connection.execute(delete(vocabulary).where(vocabulary.c.id == vocab_id))


def review_card(vocab_id: int, quality: int) -> None:
    """Apply SM-2 scheduling; quality ranges from 0 (forgot) to 5 (easy)."""
    with _engine().begin() as connection:
        card = connection.execute(select(vocabulary).where(vocabulary.c.id == vocab_id)).mappings().first()
        if card is None:
            return
        ease = max(1.3, card["ease_factor"] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        repetitions = card["repetitions"] + 1 if quality >= 3 else 0
        if quality < 3 or repetitions == 1:
            interval = 1
        elif repetitions == 2:
            interval = 6
        else:
            interval = max(1, round(card["interval_days"] * ease))
        connection.execute(update(vocabulary).where(vocabulary.c.id == vocab_id).values(
            ease_factor=ease, interval_days=interval, repetitions=repetitions,
            due_date=date.today() + timedelta(days=interval),
        ))
    log_history("flashcard", f"Ôn từ ID {vocab_id}", float(quality))


def log_history(event_type: str, details: str = "", score: float | None = None) -> None:
    with _engine().begin() as connection:
        connection.execute(insert(study_history).values(event_type=event_type, details=details, score=score))


def save_quiz_result(quiz_type: str, score: float, total: int) -> None:
    with _engine().begin() as connection:
        connection.execute(insert(quiz_results).values(quiz_type=quiz_type, score=score, total=total))
    log_history("quiz", quiz_type, score)


def get_statistics() -> dict:
    with _engine().connect() as connection:
        word_count = connection.execute(select(func.count()).select_from(vocabulary)).scalar_one()
        timestamps = [row[0] for row in connection.execute(
            select(study_history.c.created_at).order_by(study_history.c.created_at.desc())
        ).all() if row[0]]
    days = []
    for timestamp in timestamps:
        if isinstance(timestamp, str):
            days.append(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date())
        else:
            days.append(timestamp.date())
    streak, expected = 0, date.today()
    for studied_day in dict.fromkeys(days):
        if studied_day == expected:
            streak += 1
            expected -= timedelta(days=1)
        elif studied_day < expected:
            break
    return {"words": word_count, "streak": streak, "due": len(get_due_cards(999))}


def export_csv() -> str:
    rows, output = get_vocabulary(), StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value.isoformat() if hasattr(value, "isoformat") else value for key, value in row.items()})
    return output.getvalue()
