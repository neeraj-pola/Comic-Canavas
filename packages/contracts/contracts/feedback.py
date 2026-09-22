"""Preference-signal contracts — the raw material for reward model / DPO /
GRPO training, written by the feedback node."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ImagePair(BaseModel):
    day_id: str
    panel_id: int
    chosen: str
    rejected: str
    source: Literal["ab", "regenerate"]


class CaptionPair(BaseModel):
    day_id: str
    panel_id: int
    original: str
    edited: str


class Thumb(BaseModel):
    day_id: str
    panel_id: int
    value: Literal[1, -1]
