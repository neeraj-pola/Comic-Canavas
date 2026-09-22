"""Scheduler dev loop: `python -m app.main` (see Procfile.dev).

Each tick calls the four real cron entrypoints in `app/cron.py` with the
current UTC time — they no-op unless it's actually their scheduled minute
(`is_weekly_recap_time` etc.), so a 60s tick is a safe, simple way to run
them without a real cron daemon in dev.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import psycopg
from arq import create_pool
from arq.connections import RedisSettings
from psycopg.rows import dict_row

from app.config import get_settings
from app.cron import MockNotificationAdapter, export_feedback, kick_training, remind, weekly_recap

logger = logging.getLogger("comiccanvas.scheduler")

TICK_SECONDS = 60


async def tick() -> None:
    settings = get_settings()
    now = datetime.now(UTC)
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        due = remind(now, conn=conn, adapter=MockNotificationAdapter())
        if due:
            logger.info("remind: sent to %d user(s)", len(due))

        arq_pool = await create_pool(RedisSettings.from_dsn(str(settings.redis_url)))
        try:
            recapped = await weekly_recap(now, conn=conn, arq_pool=arq_pool)
            if recapped:
                logger.info("weekly_recap: enqueued for %d user(s)", len(recapped))
        finally:
            await arq_pool.aclose()

        snapshot = export_feedback(now, conn=conn)
        if snapshot:
            logger.info("export_feedback: snapshot %s", snapshot)

        conn.commit()

    if await kick_training(now):
        logger.info("kick_training: fired")


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logger.info("scheduler dev loop starting (env=%s)", settings.env)
    while True:
        try:
            await tick()
        except Exception:
            logger.exception("scheduler tick failed")
        await asyncio.sleep(TICK_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
