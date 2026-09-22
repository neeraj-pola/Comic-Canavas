"""MockImageProvider — deterministic placeholder PNGs via Storage, no
network calls, used by every test that doesn't need a real generator.
"""

from __future__ import annotations

from pathlib import Path

from storage import LocalFileStorage

from app.images.base import IdentityRef, variant_tag
from app.images.mock import MockImageProvider, _render_placeholder
from contracts import ImagePrompt


def _storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")


def _prompt(**overrides: object) -> ImagePrompt:
    defaults: dict[str, object] = {
        "panel_id": 1,
        "generator": "mock",
        "character_clause": "[IDENTITY], content expression",
        "environment_clause": "a chipped mug, morning light, a wooden table, steam",
        "positive": "[IDENTITY], content expression, a chipped mug, morning light",
        "negative": "no artist names",
        "seed": 1000,
    }
    defaults.update(overrides)
    return ImagePrompt(**defaults)


def _identity(**overrides: object) -> IdentityRef:
    defaults: dict[str, object] = {"trigger_token": "sks person"}
    defaults.update(overrides)
    return IdentityRef(**defaults)


async def test_generate_produces_n_candidates(tmp_path: Path) -> None:
    provider = MockImageProvider(storage=_storage(tmp_path))
    candidates = await provider.generate(_prompt(), n=3, identity=_identity())

    assert len(candidates) == 3
    assert all(c.panel_id == 1 for c in candidates)
    assert all(c.url.startswith("http://localhost:8000/media/") for c in candidates)


async def test_generate_gives_each_candidate_a_distinct_seed(tmp_path: Path) -> None:
    provider = MockImageProvider(storage=_storage(tmp_path))
    candidates = await provider.generate(_prompt(seed=42), n=3, identity=_identity())

    assert [c.seed for c in candidates] == [42, 43, 44]
    assert len({c.id for c in candidates}) == 3


async def test_generate_writes_real_png_bytes(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    provider = MockImageProvider(storage=storage)
    candidates = await provider.generate(_prompt(), n=1, identity=_identity())

    key = f"candidates/panel-1/{candidates[0].seed}-{variant_tag(_prompt())}.png"
    data = storage.read_path(key).read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")


async def test_generate_defaults_seed_to_zero_when_prompt_has_none(tmp_path: Path) -> None:
    provider = MockImageProvider(storage=_storage(tmp_path))
    candidates = await provider.generate(_prompt(seed=None), n=2, identity=_identity())

    assert [c.seed for c in candidates] == [0, 1]


def test_render_placeholder_is_deterministic_for_the_same_seed() -> None:
    prompt = _prompt()
    first = _render_placeholder(prompt, seed=7, resolved_positive="sks person, content")
    second = _render_placeholder(prompt, seed=7, resolved_positive="sks person, content")
    assert first == second


def test_render_placeholder_differs_across_seeds() -> None:
    prompt = _prompt()
    a = _render_placeholder(prompt, seed=7, resolved_positive="sks person, content")
    b = _render_placeholder(prompt, seed=8, resolved_positive="sks person, content")
    assert a != b


def test_estimate_usd_is_free(tmp_path: Path) -> None:
    provider = MockImageProvider(storage=_storage(tmp_path))
    assert provider.estimate_usd(n=3) == 0.0


async def test_options_that_share_a_seed_get_their_own_files_and_ids(tmp_path: Path) -> None:
    """The three options of a panel can share a seed (they differ only in their prompt); their
    keys and ids carry a tag of the prompt text so they never overwrite each other."""
    provider = MockImageProvider(storage=_storage(tmp_path))
    prompts = [_prompt(seed=7, positive=f"same scene, phrase {i}") for i in range(3)]
    made = [await provider.generate(p, n=1, identity=_identity()) for p in prompts]
    candidates = [c[0] for c in made]
    assert {c.seed for c in candidates} == {7}
    assert len({c.id for c in candidates}) == 3 and len({c.url for c in candidates}) == 3
    files = [_storage(tmp_path).get_object_by_url(c.url) for c in candidates]
    assert len(set(files)) == 3  # and the placeholders differ, since the prompts do
