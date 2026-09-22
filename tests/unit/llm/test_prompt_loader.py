"""Prompt loader — version selection, env pin, and front matter.

Uses synthetic fixture files under `tmp_path`, not real committed prompts
— the actual `extractor.v1.md` etc. belong to other modules' content, not this
loader's concern (see app/prompts/loader.py's module docstring).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.prompts.loader import (
    PromptFrontMatterError,
    PromptNotFoundError,
    PromptPinNotFoundError,
    load_prompt,
)


def _write_prompt(
    directory: Path, filename: str, *, role: str, temperature: float, body: str
) -> None:
    (directory / filename).write_text(
        f"---\nrole: {role}\ntemperature: {temperature}\n---\n{body}\n", encoding="utf-8"
    )


def test_loads_the_highest_version_by_default(tmp_path: Path) -> None:
    _write_prompt(tmp_path, "extractor.v1.md", role="extractor", temperature=0, body="v1 body")
    _write_prompt(tmp_path, "extractor.v2.md", role="extractor", temperature=0, body="v2 body")
    _write_prompt(tmp_path, "extractor.v10.md", role="extractor", temperature=0, body="v10 body")

    prompt = load_prompt("extractor", directory=tmp_path)

    # v10 > v2 numerically, even though "v10" < "v2" as a string — proves
    # version comparison is numeric, not lexicographic.
    assert prompt.version == "v10"
    assert prompt.text == "v10 body"
    assert prompt.versioned_name == "extractor.v10"


def test_env_pin_overrides_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_prompt(tmp_path, "extractor.v1.md", role="extractor", temperature=0, body="v1 body")
    _write_prompt(tmp_path, "extractor.v2.md", role="extractor", temperature=0, body="v2 body")
    monkeypatch.setenv("PROMPT_EXTRACTOR", "v1")

    prompt = load_prompt("extractor", directory=tmp_path)

    assert prompt.version == "v1"
    assert prompt.text == "v1 body"


def test_pinning_a_missing_version_errors_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_prompt(tmp_path, "extractor.v1.md", role="extractor", temperature=0, body="v1 body")
    monkeypatch.setenv("PROMPT_EXTRACTOR", "v99")

    with pytest.raises(PromptPinNotFoundError, match="v99"):
        load_prompt("extractor", directory=tmp_path)


def test_no_matching_files_errors_clearly(tmp_path: Path) -> None:
    with pytest.raises(PromptNotFoundError, match="script"):
        load_prompt("script", directory=tmp_path)


def test_front_matter_fields_are_parsed(tmp_path: Path) -> None:
    (tmp_path / "judge.v1.md").write_text(
        "---\nrole: judge\ntemperature: 0.2\nnotes: pairwise comparison\n---\nCompare A and B.\n",
        encoding="utf-8",
    )

    prompt = load_prompt("judge", directory=tmp_path)

    assert prompt.role == "judge"
    assert prompt.temperature == 0.2
    assert prompt.notes == "pairwise comparison"
    assert prompt.text == "Compare A and B."


@pytest.mark.parametrize(
    "contents",
    [
        "no front matter at all",
        "---\nrole: extractor\n(missing closing marker)\nbody text",
        "---\ntemperature: 0\n---\nmissing role",
        "---\nrole: extractor\n---\nmissing temperature",
    ],
)
def test_malformed_front_matter_errors_clearly(tmp_path: Path, contents: str) -> None:
    (tmp_path / "extractor.v1.md").write_text(contents, encoding="utf-8")
    with pytest.raises(PromptFrontMatterError):
        load_prompt("extractor", directory=tmp_path)


def test_only_files_matching_the_exact_name_are_considered(tmp_path: Path) -> None:
    # "extractor_v2" (an unrelated file with a similar name) must not be
    # mistaken for a version of "extractor".
    _write_prompt(tmp_path, "extractor.v1.md", role="extractor", temperature=0, body="real")
    (tmp_path / "extractor_extra.v9.md").write_text(
        "---\nrole: x\ntemperature: 0\n---\nunrelated\n", encoding="utf-8"
    )

    prompt = load_prompt("extractor", directory=tmp_path)

    assert prompt.version == "v1"
