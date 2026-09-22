"""Tasks 7.1/7.2: app/compose.py — the Pillow strip/story/panel composer
and the speech-bubble placement guard.
"""

from __future__ import annotations

import random
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.compose import (
    BUBBLE_WIDTH_FRAC,
    MAX_FACE_FRAC,
    Box,
    MissingPanelImageError,
    _bubble_box,
    _grid_shape,
    _overlap_area,
    chosen_candidates_by_panel,
    compose_day,
    compose_strip,
)
from contracts import Candidate, DayState, Panel, Script

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "golden" / "compose_strip_golden.png"
# Same-environment re-renders are byte-identical (measured: 0.0) since
# Pillow rasterizes the vendored TTFs itself rather than going through
# OS font rendering; this leaves headroom for any small FreeType/Pillow
# version drift across machines while staying an order of magnitude
# below the ~0.95 signal a real caption change produces (see
# test_golden_comparison_actually_discriminates).
GOLDEN_TOLERANCE = 0.5


def _synthetic_source(color: tuple[int, int, int], size: tuple[int, int] = (300, 240)) -> bytes:
    image = Image.new("RGB", size, color)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _fixture_panels() -> list[Panel]:
    return [
        Panel(
            id=1,
            beat_id="b1",
            place="desk",
            time_of_day="morning",
            expression="content",
            action="typing on a laptop",
            framing="medium",
            caption_a="Finally answered that email.",
            caption_b="",
            bubble="Send.",
        ),
        Panel(
            id=2,
            beat_id="b2",
            place="kitchen",
            time_of_day="morning",
            expression="happy",
            action="pouring coffee",
            framing="close",
            caption_a="Coffee number two, don't judge me.",
            caption_b="",
            bubble="",
        ),
        Panel(
            id=3,
            beat_id="b3",
            place="gym",
            time_of_day="evening",
            expression="tired",
            action="lifting a dumbbell",
            framing="wide",
            caption_a="Legs day. Regret is already setting in.",
            caption_b="",
            bubble="One more set!",
        ),
        Panel(
            id=4,
            beat_id="b4",
            place="bedroom",
            time_of_day="night",
            expression="content",
            action="reading a book",
            framing="medium",
            caption_a="Ended the day on a good chapter.",
            caption_b="",
            bubble="",
        ),
    ]


def _fixture_script() -> Script:
    return Script(mood="steady and a little proud", quiet_day=False, panels=_fixture_panels())


def _fixture_images() -> dict[int, bytes]:
    return {
        1: _synthetic_source((233, 150, 60)),
        2: _synthetic_source((90, 140, 200)),
        3: _synthetic_source((120, 180, 100)),
        4: _synthetic_source((70, 70, 90)),
    }


def _fixture_face_boxes() -> dict[int, Box | None]:
    return {1: (60, 30, 160, 140), 2: (80, 20, 220, 200), 3: None, 4: (100, 40, 200, 160)}


# --- compose_strip -----------------------------------------------------


def test_compose_strip_raises_on_a_missing_panel_image() -> None:
    images = _fixture_images()
    del images[2]
    with pytest.raises(MissingPanelImageError):
        compose_strip(images, _fixture_script(), day=date(2026, 1, 1))


def test_compose_strip_produces_strip_story_and_per_panel_pngs() -> None:
    composed = compose_strip(
        _fixture_images(), _fixture_script(), day=date(2026, 1, 1), face_boxes=_fixture_face_boxes()
    )

    assert Image.open(BytesIO(composed.strip_png)).size == (2160, 2160)
    assert Image.open(BytesIO(composed.story_png)).size == (1080, 1920)
    assert set(composed.panel_pngs) == {1, 2, 3, 4}


def test_compose_strip_layout_records_a_box_per_panel() -> None:
    composed = compose_strip(
        _fixture_images(), _fixture_script(), day=date(2026, 1, 1), face_boxes=_fixture_face_boxes()
    )

    panel_layouts = composed.layout["strip"]["panels"]
    assert {p["id"] for p in panel_layouts} == {1, 2, 3, 4}
    for entry in panel_layouts:
        assert len(entry["box"]) == 4


@pytest.mark.parametrize(
    ("n_panels", "expected"), [(2, (1, 2)), (3, (2, 2)), (4, (2, 2)), (6, (2, 3))]
)
def test_grid_shape_matches_claude_md_plus_the_3_panel_generalization(
    n_panels: int, expected: tuple[int, int]
) -> None:
    assert _grid_shape(n_panels) == expected


def test_compose_strip_handles_the_quiet_day_2_panel_case() -> None:
    panels = _fixture_panels()[:2]
    script = Script(mood="quiet", quiet_day=True, panels=panels)
    images = {1: _fixture_images()[1], 2: _fixture_images()[2]}

    composed = compose_strip(images, script, day=date(2026, 1, 1))

    assert len(composed.layout["strip"]["panels"]) == 2


# --- Panel layouts must fill the canvas, stay centered, and never overlap.


def _boxes(layout: dict[str, Any]) -> tuple[list[list[int]], list[int] | None, int, int]:
    strip = layout["strip"]
    panels = [entry["box"] for entry in strip["panels"]]
    return panels, strip.get("card_box"), strip["width"], strip["height"]


def _overlap(a: list[int], b: list[int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@pytest.mark.parametrize("n_panels", [2, 3, 4, 6])
def test_strip_layout_is_centered_inside_the_canvas_and_nothing_overlaps(n_panels: int) -> None:
    images = {i: _fixture_images()[(i - 1) % 4 + 1] for i in range(1, n_panels + 1)}
    panels = [
        _fixture_panels()[(i - 1) % 4].model_copy(update={"id": i}) for i in range(1, n_panels + 1)
    ]
    script = Script(mood="steady", quiet_day=n_panels < 3, panels=panels)

    composed = compose_strip(images, script, day=date(2026, 1, 1))
    boxes, card, width, height = _boxes(composed.layout)
    everything = boxes + ([card] if card else [])

    assert Image.open(BytesIO(composed.strip_png)).size == (width, height)
    left = min(b[0] for b in everything)
    right = width - max(b[2] for b in everything)
    assert abs(left - right) <= 2, "grid must be centered, not hugging the left edge"
    assert max(b[3] for b in everything) <= height
    for i, a in enumerate(everything):
        for b in everything[i + 1 :]:
            assert not _overlap(a, b)


def test_a_three_panel_day_uses_a_square_day_card_instead_of_leaving_the_canvas_empty() -> None:
    panels = _fixture_panels()[:3]
    script = Script(mood="quietly triumphant", quiet_day=False, panels=panels)
    images = {i: _fixture_images()[i] for i in (1, 2, 3)}

    composed = compose_strip(images, script, day=date(2026, 9, 16))
    boxes, card, _, height = _boxes(composed.layout)

    assert card is not None and card[2] - card[0] == card[3] - card[1]  # a square 4th cell
    assert (
        card[3] - boxes[0][1] > height * 0.85
    )  # the grid spans the canvas, not just the top third
    # the card really is drawn (mustard), not left as blank canvas
    sample = _strip_pixels(
        composed.strip_png, (card[0] + 40, card[1] + 40, card[0] + 41, card[1] + 41)
    )
    red, green, blue = (int(v) for v in sample[0, 0])
    assert red > 200 and green > 170 and blue < 120


def test_a_two_panel_day_gets_a_full_width_mood_band() -> None:
    panels = _fixture_panels()[:2]
    script = Script(mood="a quiet one", quiet_day=True, panels=panels)

    composed = compose_strip(
        {1: _fixture_images()[1], 2: _fixture_images()[2]}, script, day=date(2026, 9, 16)
    )
    boxes, card, width, height = _boxes(composed.layout)

    assert card is not None and card[1] >= max(b[3] for b in boxes)  # below the panels
    assert (
        card[2] - card[0] > width * 0.9 and card[3] > height * 0.9
    )  # spans the width, reaches the bottom


def test_the_six_panel_recap_canvas_is_only_as_tall_as_it_needs_to_be() -> None:
    images = {i: _fixture_images()[(i - 1) % 4 + 1] for i in range(1, 7)}
    panels = [_fixture_panels()[(i - 1) % 4].model_copy(update={"id": i}) for i in range(1, 7)]

    composed = compose_strip(
        images, Script(mood="week", quiet_day=False, panels=panels), day=date(2026, 9, 20)
    )
    boxes, _, width, height = _boxes(composed.layout)

    assert height < width  # no empty bottom third
    assert height - max(b[3] for b in boxes) <= 40  # a gutter, not a void


# --- bubble avoidance ----------------------------------------------------


def test_bubble_defaults_to_top_right_with_no_face_box() -> None:
    box = _bubble_box(None, (1000, 800))
    w = int(1000 * BUBBLE_WIDTH_FRAC)
    assert box[2] > 1000 - w - 10  # anchored to the right edge


@pytest.mark.parametrize("seed", range(50))
def test_bubble_never_overlaps_a_realistic_face_box(seed: int) -> None:
    """50 randomized cases. Face boxes are generated within `MAX_FACE_FRAC`
    of the cell (see compose.py's module docstring for the proof this
    bound makes a zero-overlap corner always exist) — the realistic range
    for a detected face in a medium/close panel, not an adversarial worst
    case."""
    rng = random.Random(seed)
    cell_w = rng.randint(400, 1600)
    cell_h = rng.randint(400, 1600)

    face_w = int(cell_w * rng.uniform(0.1, MAX_FACE_FRAC))
    face_h = int(cell_h * rng.uniform(0.1, MAX_FACE_FRAC))
    face_x = rng.randint(0, cell_w - face_w)
    face_y = rng.randint(0, cell_h - face_h)
    face_box: Box = (face_x, face_y, face_x + face_w, face_y + face_h)

    bubble = _bubble_box(face_box, (cell_w, cell_h))

    assert _overlap_area(bubble, face_box) == 0


def test_bubble_falls_back_to_least_overlap_when_face_exceeds_the_guarantee_bound() -> None:
    """Beyond MAX_FACE_FRAC the zero-overlap guarantee no longer applies
    (documented, not silently broken) — but it must still return
    *something*, never raise."""
    huge_face: Box = (0, 0, 999, 999)
    bubble = _bubble_box(huge_face, (1000, 1000))
    assert bubble is not None


# --- chosen_candidates_by_panel / compose_day ---------------------------


def _candidate(panel_id: int, *, chosen: bool, url: str = "") -> Candidate:
    return Candidate(
        id=f"c{panel_id}",
        panel_id=panel_id,
        url=url or f"https://example.test/panel-{panel_id}.png",
        seed=1,
        chosen=chosen,
        face_box=(10, 10, 50, 50) if chosen else None,
    )


def test_chosen_candidates_by_panel_ignores_unchosen_candidates() -> None:
    state = DayState(
        job_id="j1",
        user_id="u1",
        date=date(2026, 1, 1),
        source="text",
        candidates=[
            _candidate(1, chosen=True),
            _candidate(1, chosen=False),
            _candidate(2, chosen=True),
        ],
    )

    chosen = chosen_candidates_by_panel(state)

    assert set(chosen) == {1, 2}
    assert all(c.chosen for c in chosen.values())


class _FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_presigned(self, key: str, **kwargs: object) -> str:
        raise NotImplementedError

    def get_url(self, key: str) -> str:
        return f"https://example.test/{key}"

    def get_object_by_url(self, url: str) -> bytes:
        prefix = "https://example.test/"
        return self.objects[url[len(prefix) :]]

    def put_object(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        self.objects[key] = data
        return self.get_url(key)

    def delete_prefix(self, prefix: str) -> None:
        self.objects = {k: v for k, v in self.objects.items() if not k.startswith(prefix)}


def test_compose_day_loads_chosen_images_and_writes_every_artifact() -> None:
    storage = _FakeStorage()
    images = _fixture_images()
    candidate_urls = {}
    for panel_id, data in images.items():
        key = f"candidates/panel-{panel_id}/x.png"
        candidate_urls[panel_id] = storage.put_object(key, data, content_type="image/png")

    state = DayState(
        job_id="j1",
        user_id="u1",
        date=date(2026, 1, 1),
        source="text",
        script=_fixture_script(),
        candidates=[
            _candidate(panel_id, chosen=True, url=url) for panel_id, url in candidate_urls.items()
        ],
    )

    result = compose_day(state, storage=storage)

    assert result.strip_url is not None
    assert result.layout is not None
    prefix = "strips/u1/2026-01-01"
    assert f"{prefix}/strip.png" in storage.objects
    assert f"{prefix}/story.png" in storage.objects
    assert f"{prefix}/layout.json" in storage.objects
    for panel_id in range(1, 5):
        assert f"{prefix}/panel-{panel_id}.png" in storage.objects


def test_compose_day_requires_a_script() -> None:
    state = DayState(job_id="j1", user_id="u1", date=date(2026, 1, 1), source="text", script=None)
    with pytest.raises(ValueError, match="script"):
        compose_day(state, storage=_FakeStorage())


# --- pixel-diff golden ----------------------------------------------------


def test_compose_strip_matches_the_golden_render_within_tolerance() -> None:
    """Real font rasterization can vary a hair across FreeType/Pillow
    builds (anti-aliasing, hinting) — this compares mean per-channel
    pixel difference against a tolerance, not exact bytes."""
    composed = compose_strip(
        _fixture_images(), _fixture_script(), day=date(2026, 1, 1), face_boxes=_fixture_face_boxes()
    )
    rendered = np.asarray(Image.open(BytesIO(composed.strip_png)).convert("RGB"), dtype=np.int16)
    golden = np.asarray(Image.open(GOLDEN_PATH).convert("RGB"), dtype=np.int16)

    assert rendered.shape == golden.shape
    mean_abs_diff = np.abs(rendered - golden).mean()
    assert mean_abs_diff < GOLDEN_TOLERANCE, (
        f"mean abs pixel diff {mean_abs_diff:.3f} exceeds tolerance"
    )


def test_golden_comparison_actually_discriminates() -> None:
    """Guards against a tolerance-check test that can't fail: a
    genuinely different render (different captions) must exceed the
    same tolerance the golden test uses."""
    panels = _fixture_panels()
    panels[0] = panels[0].model_copy(update={"caption_a": "A completely different caption here."})
    different_script = Script(mood="different mood entirely", quiet_day=False, panels=panels)

    composed = compose_strip(
        _fixture_images(), different_script, day=date(2026, 1, 1), face_boxes=_fixture_face_boxes()
    )
    rendered = np.asarray(Image.open(BytesIO(composed.strip_png)).convert("RGB"), dtype=np.int16)
    golden = np.asarray(Image.open(GOLDEN_PATH).convert("RGB"), dtype=np.int16)

    mean_abs_diff = np.abs(rendered - golden).mean()
    assert mean_abs_diff > GOLDEN_TOLERANCE


def _strip_pixels(strip_png: bytes, box: tuple[int, int, int, int]) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(strip_png)).convert("RGB").crop(box), dtype=np.int16)


def test_a_header_title_replaces_the_date_in_the_strip_header() -> None:
    """The weekly recap says "The week of September 14 \u2013 20, 2026" instead of
    showing the date of the last diary entry (it read like Sunday's page)."""
    day = date(2026, 9, 20)
    boxes = _fixture_face_boxes()
    plain = compose_strip(_fixture_images(), _fixture_script(), day=day, face_boxes=boxes)
    titled = compose_strip(
        _fixture_images(),
        _fixture_script(),
        day=day,
        face_boxes=boxes,
        header_title="The week of September 14 \u2013 20, 2026",
    )

    header = (0, 0, 1400, 170)
    body = (0, 200, 2160, 2160)
    assert (
        np.abs(
            _strip_pixels(plain.strip_png, header) - _strip_pixels(titled.strip_png, header)
        ).mean()
        > 0.5
    )
    assert (
        np.abs(_strip_pixels(plain.strip_png, body) - _strip_pixels(titled.strip_png, body)).mean()
        == 0
    )
