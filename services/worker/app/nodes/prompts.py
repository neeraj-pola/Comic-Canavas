"""Prompt writer — pipeline node 5.

For each panel in the `Script`, calls `llm.structured(PanelClauses)` for a
short `character_expression_clause` (expression + any per-panel style rule)
and a long `environment_clause`, then assembles the final two-layer
`ImagePrompt` deterministically in code: `character_clause` itself, camera
words from framing, the negative-prompt standard, and generator-specific
dialect all come from the style card / look card, not the LLM call — one
retry per panel if the assembled prompt misses a hard constraint the LLM
call alone can't guarantee (token budget, >=4 concrete nouns in the
environment clause).

`character_clause` = `[IDENTITY]` + the person's look card traits
(`_describe_look_card`) + the LLM's expression fragment. The look card
portion is built in code rather than asked of the LLM, since stable facts
like gender/hair/glasses need to show up in *every* panel's prompt the same
way every time to read as "the same person" across a strip.

Seed policy: one seed family per day, derived from `job_id` plus the panel
index, so panels share lighting/mood but aren't identical — see
`seed_for_panel`.
"""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel

from app.identity_token import IDENTITY_TOKEN
from app.images.routing import ProviderName, active_provider_name
from app.llm.base import LLMProvider, Message
from app.llm.routing import resolve
from app.memory import MemoryStore, get_memory_store
from app.prompts.loader import load_prompt
from app.prompts.style_card import StyleCard, load_style_card
from contracts import DayState, ImagePrompt, LookCard, Panel

# The same type `images/routing.py` resolves the real active provider
# against, so `generator` here always matches the actual `IMAGE_PROVIDER`.
Generator = ProviderName

MAX_POSITIVE_TOKENS = 90
MIN_ENVIRONMENT_NOUNS = 4

# LookCard fields deliberately left out of the deterministic character
# clause: `signature_outfit` overlaps with the LLM's own per-panel
# "consistent outfit unless the beat says otherwise" call (task 3.1),
# so stating it here too would just create two competing sources for the
# same fact; `skin_tone` isn't a useful disambiguator to restate verbatim
# in every prompt. `gender_term`/`hair`/`glasses`/`face_shape`/
# `distinguishing` are the traits actually worth pinning down every time.
_NONE_VALUES = {"", "none", "none noted", "not enough information"}


class NoScriptError(ValueError):
    def __init__(self) -> None:
        super().__init__("write_prompts: DayState.script is None")


class PanelNotFoundError(ValueError):
    def __init__(self, panel_id: int) -> None:
        super().__init__(f"write_single_prompt: no panel with id {panel_id} in DayState.script")


class PanelClauses(BaseModel):
    character_expression_clause: str
    environment_clause: str


def _describe_look_card(look_card: LookCard | None) -> str:
    """The deterministic, code-built half of `character_clause`."""
    if look_card is None:
        return "a person"

    subject = f"a {look_card.gender_term.strip()}" if look_card.gender_term.strip() else "a person"
    if look_card.age_descriptor.strip():
        subject = (
            f"a {look_card.age_descriptor.strip()} {look_card.gender_term.strip() or 'person'}"
        )
    traits = [subject]
    if look_card.hair.strip():
        traits.append(look_card.hair.strip())
    if look_card.glasses.strip().lower() not in _NONE_VALUES:
        traits.append(look_card.glasses.strip())
    if look_card.face_shape.strip():
        traits.append(f"{look_card.face_shape.strip()} face")
    if look_card.distinguishing.strip().lower() not in _NONE_VALUES:
        traits.append(look_card.distinguishing.strip())
    return ", ".join(traits)


def _approx_token_count(text: str) -> int:
    """No real tokenizer here — word count is a fine approximation for a
    prompt-length budget check (same reasoning as mock_provider.py's
    chars/4 stand-in for cost estimation)."""
    return len(text.split())


def _significant_word_count(text: str) -> int:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return len({w for w in words if len(w) > 3})


def _camera_words(style_card: StyleCard, framing: str) -> str:
    return f"{style_card.camera[framing]}, {style_card.camera['angle']}"


def _assemble_positive(
    *, generator: Generator, character_clause: str, environment_clause: str, camera_words: str
) -> str:
    if generator in ("flux", "flux_kontext"):
        # Flux/T5-conditioned models read natural sentences better than
        # SDXL-style comma tag-lists; reference-locking and style-forcing
        # additions are applied by `FluxKontextProvider` itself, not here.
        return f"{character_clause}. {environment_clause}. {camera_words}."
    return f"{character_clause}, {environment_clause}, {camera_words}"


def _build_messages(
    *,
    prompt_text: str,
    panel: Panel,
    style_card: StyleCard,
    look_card: LookCard | None,
    taste_notes: list[str] | None = None,
) -> list[Message]:
    taste = (
        [
            "",
            "what we have learned about this person's taste (from the panels they pick). Apply it "
            "ONLY to expression intensity, lighting, colour mood and how rich the environment is. "
            "Do not change the framing, the person's identity or outfit, and never add objects "
            "that are not in the scene:",
            *[f"- {note}" for note in taste_notes],
        ]
        if taste_notes
        else []
    )
    context = "\n".join(
        [
            f"place: {panel.place}",
            f"time_of_day: {panel.time_of_day}",
            f"expression: {panel.expression}",
            f"action: {panel.action}",
            f"framing: {panel.framing}",
            "",
            "style card character rules:",
            *[f"- {r}" for r in style_card.character.rules],
            "style card environment rules:",
            *[f"- {r}" for r in style_card.environment.rules],
            "",
            # The person's stable features (gender, hair, glasses, face
            # shape, distinguishing marks) are already appended to the
            # final prompt in code — never restate them here, just the
            # expression and any panel-specific style rule (e.g. outfit).
            f"signature_outfit (for outfit-continuity rules only): {look_card.signature_outfit}"
            if look_card
            else "signature_outfit: unknown",
            *taste,
        ]
    )
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": context},
    ]


def constraint_errors(positive: str, environment_clause: str) -> list[str]:
    errors = []
    if _approx_token_count(positive) > MAX_POSITIVE_TOKENS:
        errors.append(f"positive prompt is over {MAX_POSITIVE_TOKENS} tokens")
    if positive.count(IDENTITY_TOKEN) != 1:
        errors.append(f"positive prompt must contain {IDENTITY_TOKEN} exactly once")
    if _significant_word_count(environment_clause) < MIN_ENVIRONMENT_NOUNS:
        errors.append(f"environment_clause needs at least {MIN_ENVIRONMENT_NOUNS} concrete nouns")
    return errors


def seed_for_panel(job_id: str, panel_id: int) -> int:
    """One seed family per day: deterministic for a given `job_id` (a stable
    hash, not Python's randomized-per-process `hash()`) so panels share
    lighting/mood, offset per panel so they're distinct."""
    base = int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16)
    return base + panel_id


async def _write_panel_prompt(
    *,
    panel: Panel,
    prompt_text: str,
    temperature: float,
    style_card: StyleCard,
    look_card: LookCard | None,
    provider: LLMProvider,
    model: str,
    generator: Generator,
    job_id: str,
    place_ref_used: bool,
    taste_notes: list[str] | None = None,
) -> tuple[ImagePrompt, float]:
    messages = _build_messages(
        prompt_text=prompt_text,
        panel=panel,
        style_card=style_card,
        look_card=look_card,
        taste_notes=taste_notes,
    )
    clauses, result = await provider.structured(
        messages, PanelClauses, model=model, temperature=temperature
    )
    total_usd = result.usd

    character_clause = (
        f"{IDENTITY_TOKEN}, {_describe_look_card(look_card)}, {clauses.character_expression_clause}"
    )
    camera_words = _camera_words(style_card, panel.framing)
    positive = _assemble_positive(
        generator=generator,
        character_clause=character_clause,
        environment_clause=clauses.environment_clause,
        camera_words=camera_words,
    )
    errors = constraint_errors(positive, clauses.environment_clause)

    if errors:
        retry_messages: list[Message] = [
            *messages,
            {"role": "assistant", "content": clauses.model_dump_json()},
            {"role": "user", "content": "Please redo this — " + "; ".join(errors) + "."},
        ]
        clauses, retry_result = await provider.structured(
            retry_messages, PanelClauses, model=model, temperature=temperature
        )
        total_usd += retry_result.usd
        character_clause = (
            f"{IDENTITY_TOKEN}, {_describe_look_card(look_card)}, "
            f"{clauses.character_expression_clause}"
        )
        positive = _assemble_positive(
            generator=generator,
            character_clause=character_clause,
            environment_clause=clauses.environment_clause,
            camera_words=camera_words,
        )

    negative = ", ".join(style_card.negative_standard)
    image_prompt = ImagePrompt(
        panel_id=panel.id,
        generator=generator,
        character_clause=character_clause,
        environment_clause=clauses.environment_clause,
        positive=positive,
        negative=negative,
        seed=seed_for_panel(job_id, panel.id),
        guidance=None,
        steps=None,
        place_ref_used=place_ref_used,
    )
    return image_prompt, total_usd


async def write_single_prompt(
    state: DayState,
    panel_id: int,
    *,
    memory: MemoryStore | None = None,
    generator: Generator | None = None,
    look_card: LookCard | None = None,
    taste_notes: list[str] | None = None,
) -> tuple[ImagePrompt, float]:
    """Rewrites one panel's prompt without touching any other panel's — the
    critic's retry path when a panel fails the identity gate. Does its own
    `load_prompt`/`load_style_card`/`resolve` lookups rather than sharing
    `write_prompts`' loop, since those are cheap and this keeps
    `write_prompts` itself unchanged.

    `generator` defaults to `None`, resolved to the real active
    `IMAGE_PROVIDER` at call time; pass it explicitly only to override, e.g.
    in a test."""
    if state.script is None:
        raise NoScriptError
    panel = next((p for p in state.script.panels if p.id == panel_id), None)
    if panel is None:
        raise PanelNotFoundError(panel_id)

    memory_store = memory or get_memory_store()
    prompt = load_prompt("promptwriter")
    style_card = load_style_card()
    provider, model = resolve("prompts")
    place_ref = memory_store.place_ref(state.user_id, panel.place)

    return await _write_panel_prompt(
        panel=panel,
        prompt_text=prompt.text,
        temperature=prompt.temperature,
        style_card=style_card,
        look_card=look_card,
        provider=provider,
        model=model,
        generator=generator if generator is not None else active_provider_name(),
        job_id=state.job_id,
        place_ref_used=place_ref is not None,
        taste_notes=taste_notes,
    )


async def write_prompts(
    state: DayState,
    *,
    memory: MemoryStore | None = None,
    generator: Generator | None = None,
    look_card: LookCard | None = None,
    taste_notes: list[str] | None = None,
) -> DayState:
    if state.script is None:
        raise NoScriptError

    memory_store = memory or get_memory_store()
    # Prompt *file* is "promptwriter"; LLM routing *role* is "prompts" — two
    # different names for two different registries.
    prompt = load_prompt("promptwriter")
    style_card = load_style_card()
    provider, model = resolve("prompts")
    resolved_generator = generator if generator is not None else active_provider_name()

    total_usd = 0.0
    image_prompts: list[ImagePrompt] = []
    for panel in state.script.panels:
        # Looked up here so ImagePrompt.place_ref_used records whether a
        # reference existed at prompt-writing time; the image generator
        # (task 4.6) does the same lookup again to fetch the actual
        # reference and pass it to the provider as guidance.
        place_ref = memory_store.place_ref(state.user_id, panel.place)

        image_prompt, panel_usd = await _write_panel_prompt(
            panel=panel,
            prompt_text=prompt.text,
            temperature=prompt.temperature,
            style_card=style_card,
            look_card=look_card,
            provider=provider,
            model=model,
            generator=resolved_generator,
            job_id=state.job_id,
            place_ref_used=place_ref is not None,
            taste_notes=taste_notes,
        )
        image_prompts.append(image_prompt)
        total_usd += panel_usd

    versions = {
        **state.versions,
        "promptwriter_prompt": prompt.versioned_name,
        "promptwriter_model": f"{provider.name}:{model}",
    }
    return state.model_copy(
        update={
            "prompts": image_prompts,
            "cost_usd": state.cost_usd + total_usd,
            "versions": versions,
        }
    )
