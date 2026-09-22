"""FastAPI entrypoint (services/api). Wires the real routers: `/media/*`,
`/days`/`/jobs`, `/feedback`/regenerate, `/library`/`/search`, `/weekly`,
`/cast`, `/settings`, `/learning`, `/account`, plus `/health`.

CORS middleware is required since the frontend runs on a different origin —
see `Settings.cors_allow_origin_regex`'s own comment for why it's a regex
rather than a fixed origin list in dev.
"""

from __future__ import annotations

import logging

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg import AsyncConnection
from psycopg import OperationalError as PsycopgOperationalError
from pydantic import BaseModel

from app.config import get_settings
from app.routers import account, cast, days, feedback, learning, library, media, weekly
from app.routers import settings as settings_router

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("comiccanvas.api")

app = FastAPI(title="Comic Canvas API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=get_settings().cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(media.router)
app.include_router(days.router)
app.include_router(feedback.router)
app.include_router(feedback.days_router)
app.include_router(library.router)
app.include_router(weekly.router)
app.include_router(cast.router)
app.include_router(settings_router.router)
app.include_router(learning.router)
app.include_router(account.router)


class HealthResponse(BaseModel):
    status: str
    env: str
    db: bool
    redis: bool


async def _check_db(database_url: str) -> bool:
    try:
        async with await AsyncConnection.connect(database_url, connect_timeout=3) as conn:
            await conn.execute("SELECT 1")
        return True
    except (PsycopgOperationalError, OSError) as exc:
        logger.warning("health check: database unreachable: %s", exc)
        return False


async def _check_redis(redis_url: str) -> bool:
    client = redis.from_url(redis_url, socket_connect_timeout=3)  # type: ignore[no-untyped-call]
    try:
        return bool(await client.ping())
    except (redis.RedisError, OSError) as exc:
        logger.warning("health check: redis unreachable: %s", exc)
        return False
    finally:
        await client.aclose()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    db_ok = await _check_db(settings.database_url)
    redis_ok = await _check_redis(str(settings.redis_url))
    return HealthResponse(
        status="ok" if (db_ok and redis_ok) else "degraded",
        env=settings.env,
        db=db_ok,
        redis=redis_ok,
    )
