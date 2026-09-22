"""Postgres access for the API service — raw psycopg, matching the rest of
this codebase's convention. Real tables come from Alembic migrations, not
created here; this module only opens connections.

Tests scope to a dedicated schema: set `DB_SCHEMA` and run migrations
against it first, then every `get_conn()` call picks it up automatically
(read from the environment per-call, not baked into a default argument, so
a test's `monkeypatch.setenv` still takes effect) — no route needs its own
schema plumbing, and dev/prod (`DB_SCHEMA` unset) are unaffected.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import DictRow, dict_row

from app.config import get_settings

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class InvalidSchemaNameError(ValueError):
    def __init__(self, schema: str) -> None:
        super().__init__(f"invalid schema name: {schema!r}")


def validate_schema_identifier(schema: str) -> None:
    if not _IDENTIFIER_RE.match(schema):
        raise InvalidSchemaNameError(schema)


def fetch_one(conn: psycopg.Connection[DictRow], query: str, params: tuple[object, ...]) -> DictRow:
    """For queries that always return exactly one row (`RETURNING`,
    `count(*)`) — narrows away the `DictRow | None` mypy sees on every
    `fetchone()`, since a real absence there is a bug, not a case to
    handle."""
    row = conn.execute(query, params).fetchone()
    assert row is not None, f"expected exactly one row: {query!r}"
    return row


@contextmanager
def get_conn(*, schema: str | None = None) -> Iterator[psycopg.Connection[DictRow]]:
    """Yields a dict-row connection scoped to `schema` (default: the
    `DB_SCHEMA` env var, or `"public"` if unset). Commits on success,
    rolls back on any exception — callers don't call `commit()` themselves."""
    schema = schema or os.environ.get("DB_SCHEMA", "public")
    validate_schema_identifier(schema)
    conn = psycopg.connect(get_settings().database_url, row_factory=dict_row)
    try:
        if schema != "public":
            # Keep `public` in the path so `vector`'s type stays visible
            # even though nothing here uses it directly today.
            conn.execute(f'SET search_path TO "{schema}", public')
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
