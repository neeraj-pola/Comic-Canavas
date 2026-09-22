"""The daily pipeline's contracts — one Pydantic model per pipeline node's
input/output, plus `DayState`, the LangGraph state that threads through all
of them.

`X | None` fields default to `None` for pipeline fields not yet computed by
an earlier node — e.g. a fresh `DayState` has no `beats` until node 3 runs.
Pydantic v2 treats an un-defaulted `X | None` as required-but-nullable,
which would force every node to pass placeholders for fields it hasn't
computed yet.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class Beat(BaseModel):
    id: str
    time: Literal["morning", "midday", "afternoon", "evening", "night", "unknown"]
    place: Literal["bedroom", "desk", "kitchen", "gym", "outdoors", "transit", "cafe", "other"]
    place_detail: str | None = None  # "small kitchen with a window", used for environment prompts
    event: str = Field(max_length=200)
    emotion: str
    people: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    importance: float = Field(ge=0, le=1)
    humor: float = Field(ge=0, le=1)
    quote: str | None = None  # verbatim from transcript or None


class BeatSheet(BaseModel):
    date: date
    mood_arc: list[str] = Field(min_length=1, max_length=6)
    beats: list[Beat] = Field(min_length=1, max_length=6)
    people_mentioned: list[str] = Field(default_factory=list)
    flags: list[Literal["too_short", "sensitive", "no_events"]] = Field(default_factory=list)


class Panel(BaseModel):
    id: int  # 1..4 (or 1..2 on quiet days)
    beat_id: str
    place: str
    time_of_day: str
    expression: Literal[
        "happy", "content", "sad", "panic", "angry", "tired", "surprised", "neutral"
    ]
    action: str  # one visual sentence
    framing: Literal["wide", "medium", "close"]
    caption_a: str = Field(max_length=60)
    caption_b: str = Field(max_length=60)
    bubble: str = Field(max_length=20, default="")
    cast: list[str] = Field(default_factory=list)  # person ids in frame


class Script(BaseModel):
    mood: str
    quiet_day: bool
    # max_length is 6, not 4, so the weekly recap graph can reuse this same
    # contract for its 6-panel recap `Script` rather than a parallel shape —
    # daily scripts still only ever produce 2-4 panels in practice.
    panels: list[Panel] = Field(min_length=2, max_length=6)


class ImagePrompt(BaseModel):
    panel_id: int
    generator: Literal["leonardo", "flux", "flux_kontext", "mock"]
    character_clause: str  # short, fixed, from look card
    environment_clause: str  # long, detailed: props, materials, lighting, texture
    positive: str  # assembled
    negative: str
    seed: int | None = None
    guidance: float | None = None
    steps: int | None = None
    place_ref_used: bool = False  # a per-place reference image was passed as guidance


class Candidate(BaseModel):
    id: str
    panel_id: int
    url: str
    seed: int
    # keys: identity, style, alignment, detail, reward
    scores: dict[str, float] = Field(default_factory=dict)
    face_box: tuple[int, int, int, int] | None = None
    chosen: bool = False
    rejected_reason: str | None = None
    # DINOv2-small CLS embedding of the image (384 floats), computed when the candidate is
    # scored. What the personal ranking model learns from beyond the named signals; never sent
    # to the frontend.
    embedding: list[float] | None = None


class DayState(BaseModel):  # the LangGraph state
    job_id: str
    user_id: str
    date: date
    source: Literal["text", "audio"]
    text: str | None = None
    audio_url: str | None = None
    transcript: str | None = None
    beats: BeatSheet | None = None
    script: Script | None = None
    prompts: list[ImagePrompt] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    retries: dict[int, int] = Field(default_factory=dict)
    strip_url: str | None = None
    layout: dict[str, Any] | None = None
    cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)  # prompt/model/checkpoint versions used
