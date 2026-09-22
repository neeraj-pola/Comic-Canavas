"""nodes/prompts.py — builds ImagePrompt per panel, retries once on a
constraint violation the LLM call alone can't guarantee, and derives a
deterministic per-day seed family.

`character_clause` is built deterministically from the look card
(`_describe_look_card`) rather than LLM-authored — the LLM only authors
`character_expression_clause` (expression + outfit rule). See
`nodes/prompts.py`'s module docstring for why.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.identity_token import IDENTITY_TOKEN
from app.llm import routing
from app.llm.mock_provider import MockProvider
from app.memory import InMemoryMemoryStore
from app.nodes.prompts import (
    NoScriptError,
    _describe_look_card,
    constraint_errors,
    seed_for_panel,
    write_prompts,
)
from contracts import DayState, LookCard, Panel, Script


def _resolved_mock_provider() -> MockProvider:
    provider, _ = routing.resolve("prompts")
    assert isinstance(provider, MockProvider)
    return provider


def _script() -> Script:
    return Script(
        mood="content",
        quiet_day=False,
        panels=[
            Panel(
                id=1,
                beat_id="b1",
                place="kitchen",
                time_of_day="morning",
                expression="content",
                action="Pouring coffee",
                framing="wide",
                caption_a="Morning coffee",
                caption_b="Coffee first",
            ),
            Panel(
                id=2,
                beat_id="b2",
                place="desk",
                time_of_day="evening",
                expression="happy",
                action="Closing the laptop",
                framing="close",
                caption_a="Done for the day",
                caption_b="Wrapped up",
            ),
        ],
    )


def _day_state() -> DayState:
    return DayState(job_id="job-abc-123", user_id="u1", date=date(2026, 3, 1), source="text")


def _clauses_json(expression: str, environment: str) -> str:
    import json

    return json.dumps(
        {"character_expression_clause": expression, "environment_clause": environment}
    )


def _look_card(**overrides: str) -> LookCard:
    fields = {
        "hair": "short black wavy hair",
        "glasses": "black rectangular glasses",
        "skin_tone": "medium",
        "face_shape": "oval",
        "signature_outfit": "mustard sweater",
        "distinguishing": "a small mole above the left eyebrow",
        "gender_term": "man",
    }
    fields.update(overrides)
    return LookCard(**fields)


_GOOD_ENV = (
    "a chipped ceramic mug, steam rising off the pan, a cast iron skillet, "
    "morning light through the window"
)


@pytest.fixture(autouse=True)
def _mock_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    routing.reset_provider_cache()
    monkeypatch.setenv("LLM_PROMPTS", "mock:mock-prompts")
    # `write_prompts`/`write_single_prompt` resolve the real
    # `IMAGE_PROVIDER` when `generator` isn't given explicitly. Pinned
    # here so this file's own assertions never depend on whatever a real
    # local `.env` happens to leave in `os.environ`.
    monkeypatch.setenv("IMAGE_PROVIDER", "mock")


async def test_write_prompts_produces_one_image_prompt_per_panel() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response("PanelClauses", _clauses_json("content expression", _GOOD_ENV))

    state = _day_state().model_copy(update={"script": _script()})
    result_state = await write_prompts(state, memory=InMemoryMemoryStore())

    assert len(result_state.prompts) == 2
    assert [p.panel_id for p in result_state.prompts] == [1, 2]
    assert all(p.positive.count(IDENTITY_TOKEN) == 1 for p in result_state.prompts)


async def test_write_prompts_uses_sentence_assembly_for_flux_kontext() -> None:
    """flux_kontext (fal-ai/flux-2/edit) is the same Flux/T5 model family
    as "flux" — same natural-sentence assembly, not the SDXL-style comma
    tag-list "leonardo" uses. The reference-locking and whole-scene
    style-forcing text is added later by `FluxKontextProvider` itself,
    not here."""
    provider = _resolved_mock_provider()
    provider.set_structured_response("PanelClauses", _clauses_json("content expression", _GOOD_ENV))

    state = _day_state().model_copy(update={"script": _script()})
    result_state = await write_prompts(
        state, memory=InMemoryMemoryStore(), generator="flux_kontext"
    )

    positive = result_state.prompts[0].positive
    assert positive.endswith(".")
    assert ". " in positive  # sentence-joined, not comma tag-list


async def test_write_prompts_defaults_generator_to_the_real_image_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`generator` must not default to a hardcoded `"mock"` — every real
    panel prompt, under any real `IMAGE_PROVIDER`, would otherwise use
    the wrong assembly dialect and record `ImagePrompt.generator="mock"`
    even for a real, paid generation. Not passing `generator=` at all
    (the real production call shape) must resolve to whatever
    `IMAGE_PROVIDER` actually is."""
    monkeypatch.setenv("IMAGE_PROVIDER", "flux_kontext")
    provider = _resolved_mock_provider()
    provider.set_structured_response("PanelClauses", _clauses_json("content expression", _GOOD_ENV))

    state = _day_state().model_copy(update={"script": _script()})
    result_state = await write_prompts(state, memory=InMemoryMemoryStore())

    assert all(p.generator == "flux_kontext" for p in result_state.prompts)
    assert result_state.prompts[0].positive.endswith(".")  # sentence assembly, not comma tag-list


async def test_write_prompts_bakes_look_card_traits_into_every_panel() -> None:
    """gender/hair/glasses/distinguishing show up in *every* panel's
    character_clause, the same way every time — not something the LLM
    has to remember to restate."""
    provider = _resolved_mock_provider()
    provider.set_structured_response("PanelClauses", _clauses_json("content expression", _GOOD_ENV))

    state = _day_state().model_copy(update={"script": _script()})
    result_state = await write_prompts(state, memory=InMemoryMemoryStore(), look_card=_look_card())

    for prompt in result_state.prompts:
        assert "a man" in prompt.character_clause
        assert "short black wavy hair" in prompt.character_clause
        assert "black rectangular glasses" in prompt.character_clause
        assert "mole above the left eyebrow" in prompt.character_clause
        # signature_outfit is deliberately excluded from the deterministic
        # clause (it's the LLM's own per-panel outfit-continuity call).
        assert "mustard sweater" not in prompt.character_clause


async def test_write_prompts_falls_back_to_generic_person_without_a_look_card() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response("PanelClauses", _clauses_json("content expression", _GOOD_ENV))

    state = _day_state().model_copy(update={"script": _script()})
    result_state = await write_prompts(state, memory=InMemoryMemoryStore())

    assert "a person" in result_state.prompts[0].character_clause


async def test_write_prompts_records_place_ref_used_per_panel() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response("PanelClauses", _clauses_json("content", _GOOD_ENV))

    memory = InMemoryMemoryStore()
    memory.set_place_ref("u1", "kitchen", "ref-kitchen-1")
    # "desk" (panel 2's place) deliberately has no reference set.

    state = _day_state().model_copy(update={"script": _script()})
    result_state = await write_prompts(state, memory=memory)

    assert result_state.prompts[0].place_ref_used is True
    assert result_state.prompts[1].place_ref_used is False


async def test_write_prompts_records_versions_and_cost() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response("PanelClauses", _clauses_json("content", _GOOD_ENV))

    state = _day_state().model_copy(update={"script": _script()})
    result_state = await write_prompts(state, memory=InMemoryMemoryStore())

    assert result_state.versions["promptwriter_prompt"] == "promptwriter.v2"
    assert result_state.versions["promptwriter_model"] == "mock:mock-prompts"


async def test_write_prompts_retries_when_environment_lacks_concrete_nouns() -> None:
    provider = _resolved_mock_provider()
    vague = _clauses_json("happy", "a nice place with some stuff")
    concrete = _clauses_json("happy", _GOOD_ENV)
    provider.set_structured_response("PanelClauses", [vague, concrete])

    state = _day_state().model_copy(update={"script": _script()})
    result_state = await write_prompts(state, memory=InMemoryMemoryStore())

    prompt = result_state.prompts[0]
    assert not constraint_errors(prompt.positive, prompt.environment_clause)


def test_describe_look_card_includes_gender_hair_glasses_and_distinguishing() -> None:
    described = _describe_look_card(_look_card())
    assert "a man" in described
    assert "short black wavy hair" in described
    assert "black rectangular glasses" in described
    assert "mole above the left eyebrow" in described


def test_describe_look_card_omits_none_values() -> None:
    described = _describe_look_card(
        _look_card(glasses="none", distinguishing="none noted", gender_term="")
    )
    assert "glasses" not in described
    assert "a person" in described  # no gender_term set → generic fallback


def test_describe_look_card_none_falls_back_to_generic_person() -> None:
    assert _describe_look_card(None) == "a person"


def test_describe_look_card_includes_age_descriptor_when_set() -> None:
    described = _describe_look_card(_look_card(age_descriptor="24-year-old"))
    assert "a 24-year-old man" in described


def test_describe_look_card_omits_age_when_not_set() -> None:
    described = _describe_look_card(_look_card())
    assert "a man" in described
    assert "year-old" not in described


async def test_missing_script_raises() -> None:
    with pytest.raises(NoScriptError):
        await write_prompts(_day_state(), memory=InMemoryMemoryStore())


def test_seed_is_deterministic_for_a_given_job_id() -> None:
    assert seed_for_panel("job-abc-123", 1) == seed_for_panel("job-abc-123", 1)


def test_seed_differs_per_panel_within_the_same_job() -> None:
    assert seed_for_panel("job-abc-123", 1) != seed_for_panel("job-abc-123", 2)


def test_seed_differs_across_jobs() -> None:
    assert seed_for_panel("job-abc-123", 1) != seed_for_panel("job-xyz-999", 1)


def test_constraint_errors_flags_too_many_tokens() -> None:
    long_text = f"{IDENTITY_TOKEN} " + "word " * 100
    errors = constraint_errors(long_text, _GOOD_ENV)
    assert any("tokens" in e for e in errors)


def test_constraint_errors_flags_duplicate_identity_token() -> None:
    errors = constraint_errors(f"{IDENTITY_TOKEN} {IDENTITY_TOKEN}", _GOOD_ENV)
    assert any("exactly once" in e for e in errors)


def test_constraint_errors_flags_vague_environment() -> None:
    errors = constraint_errors(f"{IDENTITY_TOKEN}, a place", "a nice spot")
    assert any("concrete nouns" in e for e in errors)


def test_constraint_errors_empty_for_a_good_prompt() -> None:
    positive = f"{IDENTITY_TOKEN}, content, {_GOOD_ENV}, wide shot"
    assert constraint_errors(positive, _GOOD_ENV) == []


async def test_taste_notes_reach_the_prompt_writer_but_not_its_framing_or_identity() -> None:
    """The written taste notes shape expression, lighting and colour mood
    in the image prompts too — the framing is the script's job."""
    from app.nodes.prompts import _build_messages
    from app.prompts.style_card import load_style_card

    panel = _script().panels[0]
    with_notes = _build_messages(
        prompt_text="system",
        panel=panel,
        style_card=load_style_card(),
        look_card=None,
        taste_notes=["Prefers calm, understated expressions."],
    )
    body = with_notes[1]["content"]
    assert isinstance(body, str)
    assert "Prefers calm, understated expressions." in body
    assert "Do not change the framing" in body
    without = _build_messages(
        prompt_text="system", panel=panel, style_card=load_style_card(), look_card=None
    )
    assert "taste" not in str(without[1]["content"])
