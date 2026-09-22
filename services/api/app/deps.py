"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import AsyncIterator

import psycopg
from arq import ArqRedis, create_pool
from arq.connections import RedisSettings
from fastapi import Depends
from psycopg.rows import DictRow
from storage import Storage

from app.auth import Identity, get_identity
from app.config import get_settings
from app.db import get_conn
from app.storage import get_storage


def upsert_user(conn: psycopg.Connection[DictRow], user_id: str, email: str | None) -> None:
    conn.execute(
        "INSERT INTO users (id, email) VALUES (%s, %s) "
        "ON CONFLICT (id) DO UPDATE SET email = COALESCE(EXCLUDED.email, users.email)",
        (user_id, email),
    )


async def current_user_id(identity: Identity = Depends(get_identity)) -> str:
    """Ensures a `users` row exists for the caller. Runs on every
    authenticated request — cheap, and idempotent by construction."""
    with get_conn() as conn:
        upsert_user(conn, identity.user_id, identity.email)
    return identity.user_id


def storage_dep() -> Storage:
    return get_storage()


_arq_pool: ArqRedis | None = None


async def get_arq_pool() -> ArqRedis:
    """One pooled connection per process, matching arq's own recommended
    usage — not per-request, since opening a fresh redis connection on
    every `POST /days` would be wasteful."""
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(RedisSettings.from_dsn(str(get_settings().redis_url)))
    return _arq_pool


async def reset_arq_pool() -> None:
    """Test teardown — a fresh pool per test avoids cross-test event-loop
    reuse errors (a pooled connection from a closed loop can't be reused
    by the next test's loop)."""
    global _arq_pool
    if _arq_pool is not None:
        await _arq_pool.aclose()
        _arq_pool = None


async def arq_pool_dep() -> AsyncIterator[ArqRedis]:
    yield await get_arq_pool()
