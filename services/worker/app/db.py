"""Postgres connection + schema bootstrap for the memory store.

`ensure_schema` creates a small `memory_`-prefixed table set scoped to what
`memory.py` needs, idempotently (`CREATE TABLE IF NOT EXISTS`) — separate
from the main Alembic-managed schema (`users`, `days`, `beats`, etc.).

Tests connect with a dedicated schema so they never touch dev data — Postgres
schemas, not separate databases, so the same `DATABASE_URL` works for both;
`search_path` scopes every query to the given schema.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings
from app.embeddings import EMBEDDING_DIM

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class InvalidSchemaNameError(ValueError):
    def __init__(self, schema: str) -> None:
        super().__init__(f"invalid schema name: {schema!r}")


def _dsn() -> str:
    """Goes through `Settings` (not raw `os.environ`) so this resolves `.env`
    the same, correct way regardless of the caller's cwd."""
    return get_settings().database_url


def validate_schema_identifier(schema: str) -> None:
    """Public so `app.graph`'s `open_day_graph` can reuse this same check for
    its own schema-scoped checkpointer DSN — every place that interpolates a
    schema name into SQL shares one validation rule."""
    if not _IDENTIFIER_RE.match(schema):
        raise InvalidSchemaNameError(schema)


@contextmanager
def connect(*, schema: str = "public") -> Iterator[psycopg.Connection]:
    """Yields a connection scoped to `schema`, with the `vector`
    extension enabled, the schema's tables ensured, and pgvector's type
    codec registered. Commits on success, rolls back on any exception."""
    validate_schema_identifier(schema)
    conn = psycopg.connect(_dsn())
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if schema != "public":
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            # Keep "public" in the search_path alongside the target schema:
            # the `vector` type lives in `public` (where the extension was
            # created), so replacing the path entirely would hide it from
            # `register_vector`'s type lookup.
            conn.execute(f'SET search_path TO "{schema}", public')
        register_vector(conn)
        ensure_schema(conn)
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS memory_beats (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            beat_id TEXT NOT NULL,
            date DATE NOT NULL,
            time TEXT NOT NULL,
            place TEXT NOT NULL,
            place_detail TEXT,
            event TEXT NOT NULL,
            emotion TEXT NOT NULL,
            people TEXT[] NOT NULL DEFAULT '{{}}',
            objects TEXT[] NOT NULL DEFAULT '{{}}',
            importance DOUBLE PRECISION NOT NULL,
            humor DOUBLE PRECISION NOT NULL,
            quote TEXT,
            panel_urls TEXT[] NOT NULL DEFAULT '{{}}',
            embedding vector({EMBEDDING_DIM}) NOT NULL,
            UNIQUE (user_id, date, beat_id)
        )
        """
    )
    # Migrates a table created under the old `UNIQUE (user_id, beat_id)`
    # constraint to `(user_id, date, beat_id)` — `beat_id` alone is only
    # unique within one day's `BeatSheet`. No-ops on a fresh install, since
    # `ensure_schema`'s own `CREATE TABLE` above already has the fixed one.
    conn.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'memory_beats_user_id_beat_id_key'
                  AND conrelid = 'memory_beats'::regclass
            ) THEN
                ALTER TABLE memory_beats DROP CONSTRAINT memory_beats_user_id_beat_id_key;
                ALTER TABLE memory_beats ADD CONSTRAINT memory_beats_user_id_date_beat_id_key
                    UNIQUE (user_id, date, beat_id);
            END IF;
        END $$;
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS memory_beats_embedding_idx "
        "ON memory_beats USING hnsw (embedding vector_cosine_ops)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_people (
            user_id TEXT NOT NULL,
            person TEXT NOT NULL,
            last_seen DATE NOT NULL,
            PRIMARY KEY (user_id, person)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_places (
            user_id TEXT NOT NULL,
            place TEXT NOT NULL,
            ref_url TEXT,
            PRIMARY KEY (user_id, place)
        )
        """
    )


def drop_schema(schema: str) -> None:
    """Test teardown helper — drops the whole dedicated test schema."""
    validate_schema_identifier(schema)
    if schema == "public":
        raise InvalidSchemaNameError(schema)
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
