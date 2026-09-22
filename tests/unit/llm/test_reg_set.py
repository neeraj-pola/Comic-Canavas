"""ml/identity/reg_set.py — regularization-set generation. Everything
network-shaped (`fal_jobs.submit_and_wait`/`download`) is monkeypatched;
these tests never make an HTTP call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.prompts.style_card import load_style_card
from ml.identity import reg_set


@pytest.fixture(autouse=True)
def _mock_fal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_submit_and_wait(
        model_id: str, body: dict[str, Any], *, client: object
    ) -> dict[str, Any]:
        return {"images": [{"url": f"https://cdn.example/{body['seed']}.jpg"}]}

    async def _fake_download(url: str, *, client: object) -> bytes:
        return b"fake-jpeg-bytes"

    monkeypatch.setattr(reg_set, "submit_and_wait", _fake_submit_and_wait)
    monkeypatch.setattr(reg_set, "download", _fake_download)


async def test_generate_reg_set_writes_jpg_and_txt_pairs(tmp_path: Path) -> None:
    out_dir = tmp_path / "comic_character"
    report = await reg_set.generate_reg_set(n=3, out_dir=out_dir, confirm=True)

    assert report.actual_images == 3
    for stem in ("000", "001", "002"):
        assert (out_dir / f"{stem}.jpg").read_bytes() == b"fake-jpeg-bytes"
        assert (out_dir / f"{stem}.txt").read_text() == "comic character"


async def test_generate_reg_set_writes_manifest_with_seed_and_prompt(tmp_path: Path) -> None:
    out_dir = tmp_path / "comic_character"
    await reg_set.generate_reg_set(n=2, out_dir=out_dir, confirm=True)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest) == 2
    assert {"index", "subject", "seed", "prompt"} <= set(manifest[0])
    assert manifest[0]["seed"] != manifest[1]["seed"]


async def test_generate_reg_set_resumes_after_existing_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "comic_character"
    await reg_set.generate_reg_set(n=3, out_dir=out_dir, confirm=True)  # writes 000-002

    await reg_set.generate_reg_set(n=2, out_dir=out_dir, confirm=True)  # should resume at 003

    written = sorted(p.stem for p in out_dir.glob("*.jpg"))
    assert written == ["000", "001", "002", "003", "004"]
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest) == 5  # both runs' entries preserved, not overwritten


async def test_generate_reg_set_raises_without_confirm(tmp_path: Path) -> None:
    with pytest.raises(reg_set.SpendNotConfirmedError):
        await reg_set.generate_reg_set(n=3, out_dir=tmp_path, confirm=False)


def test_estimate_usd_is_monotonic_in_n() -> None:
    assert reg_set.estimate_usd(150) > reg_set.estimate_usd(5) > reg_set.estimate_usd(0)


def test_build_prompt_includes_style_phrase_and_negatives() -> None:
    style = load_style_card()
    positive, negative = reg_set.build_prompt("a person reading", style)
    assert "a person reading" in positive
    assert style.style_phrase in positive
    assert style.negative_standard[0] in negative
