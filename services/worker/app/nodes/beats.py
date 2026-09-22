"""Beat extractor — pipeline node 3.

Builds messages from the transcript plus known people/places/vocab from the
memory store, calls `llm.structured(BeatSheet)`, enforces the `too_short`
safety net, and persists via `memory.write_beats`.
"""

from __future__ import annotations

from app.llm.base import Message
from app.llm.routing import resolve
from app.memory import MemoryStore, get_memory_store
from app.prompts.loader import load_prompt
from contracts import BeatSheet, DayState

MIN_BEATS_FOR_NOT_TOO_SHORT = 3


def build_messages(
    *,
    prompt_text: str,
    transcript: str,
    date_iso: str,
    known_people: list[str],
    known_places: list[str],
    known_vocab: list[str],
) -> list[Message]:
    context = "\n".join(
        [
            f"date: {date_iso}",
            f"known_people: {', '.join(known_people) or '(none yet)'}",
            f"known_places: {', '.join(known_places) or '(none yet)'}",
            f"known_vocab: {', '.join(known_vocab) or '(none yet)'}",
            "",
            "transcript:",
            transcript,
        ]
    )
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": context},
    ]


class NoTranscriptError(ValueError):
    def __init__(self) -> None:
        super().__init__("extract_beats: DayState.transcript is None")


async def extract_beats(state: DayState, *, memory: MemoryStore | None = None) -> DayState:
    if state.transcript is None:
        raise NoTranscriptError

    memory_store = memory or get_memory_store()
    prompt = load_prompt("extractor")
    provider, model = resolve("extractor")

    messages = build_messages(
        prompt_text=prompt.text,
        transcript=state.transcript,
        date_iso=state.date.isoformat(),
        known_people=memory_store.known_people(state.user_id),
        known_places=memory_store.known_places(state.user_id),
        known_vocab=memory_store.known_vocab(state.user_id),
    )

    sheet, result = await provider.structured(
        messages, BeatSheet, model=model, temperature=prompt.temperature
    )

    # Safety net: the prompt asks the model to set `too_short` itself, but
    # a run that comes in under 3 beats gets it regardless of whether the
    # model remembered to.
    if len(sheet.beats) < MIN_BEATS_FOR_NOT_TOO_SHORT and "too_short" not in sheet.flags:
        sheet = sheet.model_copy(update={"flags": [*sheet.flags, "too_short"]})

    memory_store.write_beats(state.user_id, sheet)

    versions = {
        **state.versions,
        "extractor_prompt": prompt.versioned_name,
        "extractor_model": f"{result.provider}:{result.model}",
    }
    return state.model_copy(
        update={
            "beats": sheet,
            "cost_usd": state.cost_usd + result.usd,
            "versions": versions,
        }
    )
