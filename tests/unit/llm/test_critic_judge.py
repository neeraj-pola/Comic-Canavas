"""The vision judge: one call per panel over all its candidates, ratings 1-5 mapped
to 0-1, and never a reason to lose a strip."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from storage import LocalFileStorage

from app.critic.judge import (
    DIMENSIONS,
    PanelJudgement,
    build_messages,
    judge_panel,
    normalise,
)
from app.llm import routing
from app.llm.mock_provider import MockProvider
from contracts import Candidate, LookCard, Panel


def _png(colour: tuple[int, int, int]) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 64), colour).save(buf, format="PNG")
    return buf.getvalue()


def _storage(tmp_path: Path, n: int) -> tuple[LocalFileStorage, list[Candidate], str]:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    candidates = []
    for i in range(n):
        url = storage.put_object(
            f"candidates/panel-1/{i}.png", _png((i * 60, 30, 30)), content_type="image/png"
        )
        candidates.append(Candidate(id=f"c{i}", panel_id=1, url=url, seed=1, scores={}))
    reference = storage.put_object(
        "people/me/master.png", _png((200, 200, 200)), content_type="image/png"
    )
    return storage, candidates, reference


def _panel() -> Panel:
    return Panel(
        id=1,
        beat_id="b1",
        place="gym",
        time_of_day="evening",
        expression="content",
        action="Doing a deadlift",
        framing="wide",
        caption_a="Leg day",
        caption_b="Leg day!",
    )


@pytest.fixture(autouse=True)
def _mock_critic(monkeypatch: pytest.MonkeyPatch) -> None:
    routing.reset_provider_cache()
    monkeypatch.setenv("LLM_CRITIC", "mock:mock-critic")


def _judgement(*rows: tuple[str, int, int, int, int, int]) -> str:
    return PanelJudgement.model_validate(
        {"candidates": [dict(zip(("label", *DIMENSIONS), row, strict=True)) for row in rows]}
    ).model_dump_json()


def test_ratings_map_one_to_five_onto_zero_to_one() -> None:
    assert [normalise(r) for r in (1, 2, 3, 4, 5)] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_the_critic_role_has_a_default_route_so_an_existing_env_file_keeps_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_CRITIC", raising=False)
    assert routing.parse_spec(routing.DEFAULT_SPECS["critic"]) == (
        "anthropic",
        "claude-haiku-4-5-20251001",
    )
    assert "critic" in routing.validate_routing(["critic"])


def test_the_prompt_shows_the_reference_first_then_every_candidate_labelled() -> None:
    messages = build_messages(_panel(), None, ["img-a", "img-b", "img-c"], "ref")
    parts = messages[1]["content"]
    assert isinstance(parts, list)
    kinds = [p["type"] for p in parts]
    assert kinds.count("image") == 4  # the reference and three candidates
    texts = [p.get("text", "") for p in parts if p["type"] == "text"]
    assert "Doing a deadlift" in texts[0] and "Reference of the character:" in texts
    assert [t for t in texts if t.startswith("Candidate")] == [
        "Candidate A:",
        "Candidate B:",
        "Candidate C:",
    ]
    assert "colour warmth" in str(messages[0]["content"])  # told to ignore what the person picks


def test_the_outfit_from_the_look_card_is_given_only_when_it_is_known() -> None:
    def card(outfit: str) -> LookCard:
        return LookCard(
            hair="short dark hair",
            glasses="none",
            skin_tone="tan",
            face_shape="round",
            signature_outfit=outfit,
            distinguishing="none",
        )

    known, unknown = card("a red t-shirt"), card("not enough information")
    assert "a red t-shirt" in str(build_messages(_panel(), known, ["x"], None)[1]["content"])
    assert "usual outfit" not in str(build_messages(_panel(), unknown, ["x"], None)[1]["content"])


async def test_judge_panel_returns_scores_per_candidate(tmp_path: Path) -> None:
    storage, candidates, reference = _storage(tmp_path, 3)
    provider, _ = routing.resolve("critic")
    assert isinstance(provider, MockProvider)
    provider.set_structured_response(
        "PanelJudgement",
        _judgement(("A", 5, 4, 4, 5, 5), ("B", 2, 4, 3, 5, 2), ("C", 1, 3, 2, 5, 1)),
    )

    scores, usd = await judge_panel(
        _panel(), candidates, storage=storage, look_card=None, reference_url=reference
    )

    assert set(scores) == {"c0", "c1", "c2"}
    assert scores["c0"]["judge_scene"] == 1.0 and scores["c2"]["judge_scene"] == 0.0
    assert set(scores["c1"]) == {f"judge_{d}" for d in DIMENSIONS}
    assert usd >= 0.0


async def test_a_judge_failure_costs_the_strip_nothing(tmp_path: Path) -> None:
    storage, candidates, reference = _storage(tmp_path, 3)
    provider, _ = routing.resolve("critic")
    assert isinstance(provider, MockProvider)
    provider.set_structured_response("PanelJudgement", "this is not json")

    scores, usd = await judge_panel(
        _panel(), candidates, storage=storage, look_card=None, reference_url=reference
    )

    assert scores == {} and usd == 0.0


async def test_a_missing_image_is_a_failure_not_a_crash(tmp_path: Path) -> None:
    storage, candidates, _ = _storage(tmp_path, 2)
    broken = candidates[0].model_copy(update={"url": "http://localhost:8000/media/nope.png"})
    scores, _ = await judge_panel(
        _panel(), [broken, candidates[1]], storage=storage, look_card=None, reference_url=None
    )
    assert scores == {}
