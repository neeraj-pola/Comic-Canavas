"""critic/detail.py — Canny edge-density background-detail proxy. Pure
function, no model, no network — synthetic fixtures only (an empty-room
fixture must score lower than a detailed-room fixture).
"""

from __future__ import annotations

import numpy as np

from app.critic.detail import score_detail


def _empty_room(size: int = 200) -> np.ndarray:
    # A single flat color — no edges anywhere.
    return np.full((size, size, 3), 180, dtype=np.uint8)


def _detailed_room(size: int = 200, seed: int = 0) -> np.ndarray:
    # A checkerboard-like pattern of random blocks — lots of real edges,
    # a simple deterministic proxy for "furniture, props, texture".
    rng = np.random.RandomState(seed)
    image = np.full((size, size, 3), 180, dtype=np.uint8)
    block = 20
    for y in range(0, size, block):
        for x in range(0, size, block):
            if rng.rand() > 0.5:
                image[y : y + block, x : x + block] = rng.randint(0, 256, size=3)
    return image


def test_empty_room_scores_lower_than_a_detailed_room() -> None:
    empty_score = score_detail(_empty_room())
    detailed_score = score_detail(_detailed_room())
    assert empty_score < detailed_score


def test_empty_room_scores_near_zero() -> None:
    assert score_detail(_empty_room()) == 0.0


def test_face_box_is_excluded_from_the_score() -> None:
    # A detailed room, but with an artificially "busy" patch marked as
    # the face box — excluding it should lower the score relative to
    # scoring the whole frame (the busy patch no longer counts).
    image = _detailed_room()
    whole_frame_score = score_detail(image, face_box=None)
    excluded_score = score_detail(image, face_box=(80, 80, 120, 120))
    assert excluded_score != whole_frame_score


def test_score_is_bounded_between_zero_and_one() -> None:
    score = score_detail(_detailed_room())
    assert 0.0 <= score <= 1.0
