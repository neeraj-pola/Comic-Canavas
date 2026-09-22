"""Builds the weekly recap by having a language model pick highlights from the week's existing
daily panels — no images are generated, only laid out in the six-panel grid."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm.base import Message
from app.llm.routing import resolve

TARGET_PANELS = 6
MAX_PER_DAY = 2


@dataclass(frozen=True)
class SourcePanel:
    """One panel of one of the week's daily strips, with the image the person kept for it."""

    date: str  # YYYY-MM-DD
    day_mood: str
    panel_id: int
    caption: str
    action: str
    place: str
    time_of_day: str
    expression: str
    framing: str
    url: str


class Highlight(BaseModel):
    ref: int = Field(description="the number in square brackets in the list")
    caption: str = Field(max_length=60, description="short caption; may reuse the panel's own")


class WeekHighlights(BaseModel):
    mood: str = Field(max_length=40, description="two to four words describing the whole week")
    highlights: list[Highlight] = Field(min_length=1, max_length=TARGET_PANELS)


SYSTEM_PROMPT = (
    "You choose the highlights of one person's week from their daily comic panels, for a "
    "one-page weekly recap.\n\n"
    "You get every panel of the week, each with a number in square brackets, its day, that "
    "day's mood, its caption and the scene. Choose exactly the number asked for.\n"
    "- Tell the week as a whole: a high, a low, a turning point, something funny, something "
    "ordinary that says who this person is. Do not only pick the happiest moments.\n"
    f"- At most {MAX_PER_DAY} panels from any one day, and spread the choices across the week.\n"
    "- Prefer a panel with one clear moment over a vague one.\n"
    "- For each, give a caption of at most 60 characters. Reuse the panel's own caption unless "
    "a shorter or clearer one fits the recap better. Never invent events that are not in the "
    "panels.\n"
    "- Give the week a mood of two to four words."
)


def build_messages(panels: list[SourcePanel], target: int) -> list[Message]:
    lines = [
        f"[{i}] {p.date} (mood: {p.day_mood or 'steady'}) panel {p.panel_id} — "
        f'caption: "{p.caption}" — scene: {p.action} ({p.place}, {p.time_of_day}, '
        f"{p.expression})"
        for i, p in enumerate(panels)
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Choose {target} highlights from these {len(panels)} panels:\n\n"
            + "\n".join(lines),
        },
    ]


def _spread(count: int, k: int) -> list[int]:
    """`k` indices spread evenly over `count` items (used when the model's answer falls short)."""
    if k >= count:
        return list(range(count))
    return sorted({round(i * (count - 1) / max(k - 1, 1)) for i in range(k)})


def validate(
    panels: list[SourcePanel], picked: WeekHighlights, target: int
) -> list[tuple[SourcePanel, str]]:
    """Keep the model's valid, distinct picks (within the per-day limit), then top up evenly from
    the rest so the recap always has `target` panels when that many exist. Chronological."""
    chosen: dict[int, str] = {}
    per_day: dict[str, int] = {}
    for h in picked.highlights:
        if not 0 <= h.ref < len(panels) or h.ref in chosen:
            continue
        day = panels[h.ref].date
        if per_day.get(day, 0) >= MAX_PER_DAY:
            continue
        chosen[h.ref] = h.caption.strip() or panels[h.ref].caption
        per_day[day] = per_day.get(day, 0) + 1
        if len(chosen) == target:
            break
    if len(chosen) < target:
        for i in _spread(len(panels), target):
            if i in chosen:
                continue
            day = panels[i].date
            if per_day.get(day, 0) >= MAX_PER_DAY:
                continue
            chosen[i] = panels[i].caption
            per_day[day] = per_day.get(day, 0) + 1
            if len(chosen) == target:
                break
    if len(chosen) < target:  # still short (e.g. one busy day): any remaining panel will do
        for i, p in enumerate(panels):
            if len(chosen) == target:
                break
            chosen.setdefault(i, p.caption)
    return [
        (panels[i], chosen[i])
        for i in sorted(chosen, key=lambda i: (panels[i].date, panels[i].panel_id))
    ]


async def choose_highlights(
    panels: list[SourcePanel], *, target: int = TARGET_PANELS
) -> tuple[list[tuple[SourcePanel, str]], str, float]:
    """`([(panel, caption), ...] oldest first, the week's mood, usd)`. Raises if the week has no
    panels to choose from."""
    if not panels:
        raise NoPanelsInWeekError
    want = min(target, len(panels))
    provider, model = resolve("script")
    picked, meta = await provider.structured(
        build_messages(panels, want), WeekHighlights, model=model, temperature=0.3
    )
    return validate(panels, picked, want), picked.mood.strip() or "a full week", meta.usd


class NoPanelsInWeekError(ValueError):
    def __init__(self) -> None:
        super().__init__("no comic panels this week — make some days first")
