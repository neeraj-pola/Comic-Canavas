"""Script agent — pipeline node 4.

Builds messages from the `BeatSheet` + `humor_level` + few-shot examples
(`memory.top_rated_scripts`), calls `llm.structured(Script)`, and retries
once if a full day gets too few panels or the panels lack framing diversity
(at least one `wide`, one `close`) — a cross-panel constraint the schema
itself can't express, checked here instead.
"""

from __future__ import annotations

from app.llm.base import Message
from app.llm.routing import resolve
from app.memory import MemoryStore, ScriptExample, get_memory_store
from app.prompts.loader import load_prompt
from contracts import Beat, BeatSheet, DayState, Script

MAX_FEW_SHOT_EXAMPLES = 10
DEFAULT_HUMOR_LEVEL = 5
DEFAULT_MAX_PANELS = 4
FULL_DAY_PANELS = 4  # what a day with enough beats should draw


class NoBeatsError(ValueError):
    def __init__(self) -> None:
        super().__init__("write_script: DayState.beats is None")


def _format_beat(b: Beat) -> str:
    return (
        f"- id={b.id} time={b.time} place={b.place} place_detail={b.place_detail!r} "
        f"event={b.event!r} emotion={b.emotion} people={b.people} objects={b.objects} "
        f"importance={b.importance} humor={b.humor} quote={b.quote!r}"
    )


def _format_beat_sheet(beat_sheet: BeatSheet) -> str:
    lines = [
        f"date: {beat_sheet.date.isoformat()}",
        f"flags: {', '.join(beat_sheet.flags) or '(none)'}",
        "beats:",
        *[_format_beat(b) for b in beat_sheet.beats],
    ]
    return "\n".join(lines)


def _format_example(example: ScriptExample) -> str:
    panels = "\n".join(
        f"  - beat_id={p.beat_id} framing={p.framing} expression={p.expression} "
        f"action={p.action!r} caption_a={p.caption_a!r} caption_b={p.caption_b!r} "
        f"bubble={p.bubble!r}"
        for p in example.script.panels
    )
    return (
        f"Beats:\n{_format_beat_sheet(example.beat_sheet)}\n"
        f"Good script (mood={example.script.mood}, quiet_day={example.script.quiet_day}):\n{panels}"
    )


def build_messages(
    *,
    prompt_text: str,
    beat_sheet: BeatSheet,
    humor_level: int,
    examples: list[ScriptExample],
    max_panels: int = DEFAULT_MAX_PANELS,
    taste_notes: list[str] | None = None,
) -> list[Message]:
    context_parts = [
        f"humor_level: {humor_level}",
        f"max_panels: {max_panels}",
        "",
        _format_beat_sheet(beat_sheet),
    ]
    if taste_notes:
        # Written by `app.preference.notes` from this person's own picks and caption edits.
        context_parts.append(
            "\nWhat we have learned about this person's taste (from the panels they pick and the "
            "captions they edit). Let it guide framing, expression and caption tone; never let it "
            "add events that are not in the beats:"
        )
        context_parts.extend(f"- {note}" for note in taste_notes)
    if examples:
        context_parts.append(
            "\nHere are some of this user's highest-rated past scripts, for style reference:"
        )
        context_parts.extend(f"\n{_format_example(example)}" for example in examples)
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": "\n".join(context_parts)},
    ]


def needs_a_full_day(beat_sheet: BeatSheet, script: Script) -> bool:
    """A day with four or more beats and no flags (not too short, sensitive, or empty) should
    draw four panels; the writer sometimes calls such a day "quiet" and draws fewer anyway. One
    retry asking for four panels usually fixes it; if it still returns fewer, that answer stands."""
    return (
        len(beat_sheet.beats) >= FULL_DAY_PANELS
        and not beat_sheet.flags
        and len(script.panels) < FULL_DAY_PANELS
    )


def has_framing_diversity(script: Script) -> bool:
    framings = {p.framing for p in script.panels}
    return "wide" in framings and "close" in framings


async def write_script(
    state: DayState,
    *,
    memory: MemoryStore | None = None,
    humor_level: int = DEFAULT_HUMOR_LEVEL,
    max_panels: int = DEFAULT_MAX_PANELS,
    taste_notes: list[str] | None = None,
) -> DayState:
    """`max_panels` lets `weekly_graph.py` reuse this function for a 6-panel
    recap instead of the daily default of 4 — threaded into the prompt
    rather than hardcoded, so the model is told the correct ceiling."""
    if state.beats is None:
        raise NoBeatsError

    memory_store = memory or get_memory_store()
    prompt = load_prompt("script")
    provider, model = resolve("script")
    examples = memory_store.top_rated_scripts(state.user_id, n=MAX_FEW_SHOT_EXAMPLES)

    messages = build_messages(
        prompt_text=prompt.text,
        beat_sheet=state.beats,
        humor_level=humor_level,
        examples=examples,
        max_panels=max_panels,
        taste_notes=taste_notes,
    )

    script, result = await provider.structured(
        messages, Script, model=model, temperature=prompt.temperature
    )
    total_usd = result.usd

    if needs_a_full_day(state.beats, script):
        more_messages: list[Message] = [
            *messages,
            {"role": "assistant", "content": script.model_dump_json()},
            {
                "role": "user",
                "content": (
                    f"This day has {len(state.beats.beats)} distinct beats and no flags, so it is "
                    f"not a quiet day. Redo it with {FULL_DAY_PANELS} panels (quiet_day false), "
                    "one for each of four different beats — do not merge them or skip any into "
                    "a two-panel day. Keep the tone, captions and framing rules the same."
                ),
            },
        ]
        script, more_result = await provider.structured(
            more_messages, Script, model=model, temperature=prompt.temperature
        )
        total_usd += more_result.usd
        result = more_result

    if len(script.panels) >= 2 and not has_framing_diversity(script):
        retry_messages: list[Message] = [
            *messages,
            {"role": "assistant", "content": script.model_dump_json()},
            {
                "role": "user",
                "content": (
                    "Please redo this with at least one 'wide' framing and at least one "
                    "'close' framing among the panels — keep everything else the same."
                ),
            },
        ]
        script, retry_result = await provider.structured(
            retry_messages, Script, model=model, temperature=prompt.temperature
        )
        total_usd += retry_result.usd
        result = retry_result

    versions = {
        **state.versions,
        "script_prompt": prompt.versioned_name,
        "script_model": f"{result.provider}:{result.model}",
    }
    return state.model_copy(
        update={
            "script": script,
            "cost_usd": state.cost_usd + total_usd,
            "versions": versions,
        }
    )
