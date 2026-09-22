"""The style card loads and is well-formed."""

from __future__ import annotations

from app.prompts.style_card import StyleCard, load_style_card


def test_style_card_loads_and_validates() -> None:
    card = load_style_card()
    assert isinstance(card, StyleCard)


def test_negative_standard_forbids_artist_names() -> None:
    card = load_style_card()
    assert any("artist name" in rule for rule in card.negative_standard)


def test_negative_standard_forbids_photorealism() -> None:
    card = load_style_card()
    assert any("photorealism" in rule for rule in card.negative_standard)


def test_character_and_environment_rules_are_non_empty() -> None:
    card = load_style_card()
    assert len(card.character.rules) >= 3
    assert len(card.environment.rules) >= 3


def test_version_and_style_phrase_are_set() -> None:
    card = load_style_card()
    assert card.version
    assert card.style_phrase


def test_style_phrase_names_the_v2_visual_language() -> None:
    """Regularization renders and master candidates both build their
    prompt from this string, so it has to actually say what the style
    is, not just be non-empty."""
    phrase = load_style_card().style_phrase.lower()
    for word in ("flat ink", "bold", "outline", "cel shading", "halftone"):
        assert word in phrase


def test_camera_has_all_four_framing_keys() -> None:
    """Regression guard: nodes/prompts.py indexes `style_card.camera` by
    framing name plus `angle` — a style-card edit that drops one of these
    keys would only fail at prompt-writing time, not at load time,
    without this check."""
    card = load_style_card()
    assert {"wide", "medium", "close", "angle"} <= set(card.camera)
