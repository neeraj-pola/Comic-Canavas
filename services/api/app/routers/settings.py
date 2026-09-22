"""`GET/PUT /settings`, backed by the real `settings` table."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.db import get_conn
from app.deps import current_user_id

router = APIRouter(prefix="/settings", tags=["settings"])

_DEFAULTS = {
    "style": None,
    "humor": 5,
    "caption_length": None,
    "sensitive_mode": True,
    "reminder_time": None,
    "channel": None,
    "generator": "mock",
    "detail_level": None,
    "personalise": True,
}


class SettingsIn(BaseModel):
    style: str | None = None
    humor: int = 5
    caption_length: str | None = None
    sensitive_mode: bool = True
    reminder_time: str | None = None
    channel: str | None = None
    generator: str = "mock"
    detail_level: str | None = None
    # Off = what the app has learned about your taste no longer steers prompts or scripts
    # (it keeps learning from your picks, so switching back on is instant).
    personalise: bool = True


@router.get("")
async def get_user_settings(user_id: str = Depends(current_user_id)) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT style, humor, caption_length, sensitive_mode, reminder_time, "
            "channel, generator, detail_level, personalise FROM settings WHERE user_id = %s",
            (user_id,),
        ).fetchone()
    return dict(row) if row is not None else dict(_DEFAULTS)


@router.put("")
async def put_user_settings(
    body: SettingsIn, user_id: str = Depends(current_user_id)
) -> dict[str, Any]:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO settings (
                user_id, style, humor, caption_length, sensitive_mode,
                reminder_time, channel, generator, detail_level, personalise, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (user_id) DO UPDATE SET
                style = EXCLUDED.style,
                humor = EXCLUDED.humor,
                caption_length = EXCLUDED.caption_length,
                sensitive_mode = EXCLUDED.sensitive_mode,
                reminder_time = EXCLUDED.reminder_time,
                channel = EXCLUDED.channel,
                generator = EXCLUDED.generator,
                detail_level = EXCLUDED.detail_level,
                personalise = EXCLUDED.personalise,
                updated_at = now()
            """,
            (
                user_id,
                body.style,
                body.humor,
                body.caption_length,
                body.sensitive_mode,
                body.reminder_time,
                body.channel,
                body.generator,
                body.detail_level,
                body.personalise,
            ),
        )
    return body.model_dump()
