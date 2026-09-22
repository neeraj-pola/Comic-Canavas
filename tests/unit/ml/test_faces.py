"""ml/identity/faces.py — classification logic (usable/blur/
duplicate/no_face/multi_face), tested against a fake `FaceDetector`
rather than the real ~280MB buffalo_l model (same "mock the dependency,
not the transport" approach as the LLM/image provider tests) — this
module's actual detection/alignment was live-verified separately against
a real set of photos, including a real EXIF-orientation edge case (see
the module's own docstring).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.identity.faces import (
    BLUR_VARIANCE_THRESHOLD,
    DetectedFace,
    FacePhoto,
    _blur_variance,
    _largest_face,
    decode_jpeg_bgr,
    detect_and_embed,
    detect_face_box,
    encode_crop_jpeg,
    face_bbox_from_kps,
    largest_face_kps,
    process_batch,
)

# Plausible (not anatomically exact — doesn't need to be, `norm_crop` just
# needs 5 non-degenerate points to compute a similarity transform) eye/
# nose/mouth landmark positions within the synthetic images below.
_KPS = np.array([[70, 70], [130, 70], [100, 100], [75, 140], [125, 140]], dtype=np.float32)

# A visibly smaller landmark spread than _KPS, for _largest_face tests.
_SMALL_KPS = np.array([[95, 95], [105, 95], [100, 100], [96, 105], [104, 105]], dtype=np.float32)


class _FakeFace:
    def __init__(self, kps: np.ndarray) -> None:
        self.kps = kps
        self.embedding = np.ones(512, dtype=np.float32)


class _FakeDetector:
    """Returns one canned result per call, in order — mirrors
    `unittest.mock.Mock(side_effect=[...])` but keeps `FaceDetector`'s
    Protocol typing intact instead of needing `# type: ignore` at every
    call site."""

    def __init__(self, results: list[list[DetectedFace]]) -> None:
        self._results = iter(results)

    def get(self, img: np.ndarray) -> list[DetectedFace]:
        return next(self._results)


def _sharp_image(seed: int = 0, size: int = 200) -> np.ndarray:
    # High-frequency noise: a good, deterministic proxy for "sharp" —
    # strong edges everywhere, so Laplacian variance is high regardless
    # of exactly which 512x512 region `norm_crop` warps out of it.
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)


def _flat_image(size: int = 200, value: int = 128) -> np.ndarray:
    # A uniform image has ~zero second derivative everywhere — the
    # opposite extreme from _sharp_image, a reliable "blurry" proxy.
    return np.full((size, size, 3), value, dtype=np.uint8)


def _dummy_paths(tmp_path: Path, n: int) -> list[Path]:
    # Never actually decoded in these tests — load_bgr is patched to
    # return canned arrays instead, so these files just need to exist
    # and give process_batch something to attach to FacePhoto.path.
    paths = []
    for i in range(n):
        p = tmp_path / f"photo_{i}.jpg"
        p.write_bytes(b"not a real image, never decoded in these tests")
        paths.append(p)
    return paths


def _process_with_fake(
    paths: list[Path],
    images: list[np.ndarray],
    face_results: list[list[DetectedFace]],
    monkeypatch: pytest.MonkeyPatch,
) -> list[FacePhoto]:
    import ml.identity.faces as faces_module

    image_iter = iter(images)
    monkeypatch.setattr(faces_module, "load_bgr", lambda _path: next(image_iter))
    return process_batch(paths, detector=_FakeDetector(face_results))


def test_no_face_when_detector_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _dummy_paths(tmp_path, 1)
    results = _process_with_fake(paths, [_flat_image()], [[]], monkeypatch)
    assert results[0].status == "no_face"
    assert results[0].crop_bgr is None


def test_multi_face_when_detector_returns_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _dummy_paths(tmp_path, 1)
    two_faces: list[DetectedFace] = [_FakeFace(_KPS), _FakeFace(_KPS)]
    results = _process_with_fake(paths, [_sharp_image()], [two_faces], monkeypatch)
    assert results[0].status == "multi_face"
    assert results[0].crop_bgr is None


def test_usable_for_a_single_sharp_face(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _dummy_paths(tmp_path, 1)
    results = _process_with_fake(paths, [_sharp_image(seed=1)], [[_FakeFace(_KPS)]], monkeypatch)
    photo = results[0]
    assert photo.status == "usable"
    assert photo.blur_score is not None
    assert photo.blur_score >= BLUR_VARIANCE_THRESHOLD
    assert photo.embedding is not None
    assert photo.crop_bgr is not None
    assert photo.crop_bgr.shape == (512, 512, 3)


def test_blur_for_a_flat_low_variance_face(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _dummy_paths(tmp_path, 1)
    results = _process_with_fake(paths, [_flat_image()], [[_FakeFace(_KPS)]], monkeypatch)
    photo = results[0]
    assert photo.status == "blur"
    assert photo.blur_score is not None
    assert photo.blur_score < BLUR_VARIANCE_THRESHOLD
    # embedding is only kept for genuinely usable photos:
    assert photo.embedding is None


def test_second_near_identical_photo_is_marked_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _dummy_paths(tmp_path, 2)
    same_image = _sharp_image(seed=2)
    results = _process_with_fake(
        paths,
        [same_image, same_image.copy()],
        [[_FakeFace(_KPS)], [_FakeFace(_KPS)]],
        monkeypatch,
    )
    assert results[0].status == "usable"
    assert results[1].status == "duplicate"


def test_two_distinct_sharp_photos_are_both_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _dummy_paths(tmp_path, 2)
    results = _process_with_fake(
        paths,
        [_sharp_image(seed=10), _sharp_image(seed=20)],
        [[_FakeFace(_KPS)], [_FakeFace(_KPS)]],
        monkeypatch,
    )
    assert results[0].status == "usable"
    assert results[1].status == "usable"


def test_blur_variance_is_higher_for_sharp_than_flat_images() -> None:
    assert _blur_variance(_sharp_image()) > _blur_variance(_flat_image())


def test_encode_crop_jpeg_round_trips_to_valid_jpeg_bytes() -> None:
    data = encode_crop_jpeg(_sharp_image(size=64))
    assert data.startswith(b"\xff\xd8")  # JPEG magic bytes


# --- detect_and_embed / decode_jpeg_bgr (no blur/duplicate gate) ---


def test_largest_face_picks_the_bigger_landmark_spread() -> None:
    small, big = _FakeFace(_SMALL_KPS), _FakeFace(_KPS)
    assert _largest_face([small, big]) is big
    assert _largest_face([big, small]) is big  # order-independent


def test_detect_and_embed_returns_none_when_no_face() -> None:
    detector = _FakeDetector([[]])
    assert detect_and_embed(_flat_image(), detector=detector) is None


def test_detect_and_embed_bypasses_the_blur_gate() -> None:
    """The exact bug this function exists for: process_batch would flag
    this flat/low-variance image as "blur" (see
    test_blur_for_a_flat_low_variance_face above) and drop its embedding
    — detect_and_embed must still return a full result for the identical
    image, since a generated render being flagged "blur" here is a false
    positive from a gate calibrated for real camera photos, not AI
    output."""
    detector = _FakeDetector([[_FakeFace(_KPS)]])
    result = detect_and_embed(_flat_image(), detector=detector)
    assert result is not None
    assert result.crop_bgr.shape == (512, 512, 3)
    assert result.embedding is not None
    assert result.face_count == 1


def test_detect_and_embed_reports_face_count_for_multi_face_images() -> None:
    detector = _FakeDetector([[_FakeFace(_SMALL_KPS), _FakeFace(_KPS)]])
    result = detect_and_embed(_sharp_image(), detector=detector)
    assert result is not None
    assert result.face_count == 2  # caller decides whether that's disqualifying


def test_decode_jpeg_bgr_round_trips_with_encode_crop_jpeg() -> None:
    original = _sharp_image(size=64)
    decoded = decode_jpeg_bgr(encode_crop_jpeg(original))
    assert decoded.shape == original.shape


def test_decode_jpeg_bgr_raises_on_garbage_bytes() -> None:
    with pytest.raises(ValueError):
        decode_jpeg_bgr(b"not an image")


# --- largest_face_kps / face_bbox_from_kps / detect_face_box ---


def test_largest_face_kps_returns_none_without_a_detected_face() -> None:
    detector = _FakeDetector([[]])
    assert largest_face_kps(_sharp_image(), detector=detector) is None


def test_largest_face_kps_returns_the_largest_faces_landmarks() -> None:
    detector = _FakeDetector([[_FakeFace(_SMALL_KPS), _FakeFace(_KPS)]])
    kps = largest_face_kps(_sharp_image(), detector=detector)
    assert kps is not None
    np.testing.assert_array_equal(kps, _KPS)


def test_face_bbox_from_kps_is_centered_and_clamped_to_the_frame() -> None:
    kps = np.array([[145, 145], [165, 145], [155, 155], [147, 165], [163, 165]], dtype=np.float32)
    x1, y1, x2, y2 = face_bbox_from_kps((300, 300), kps)
    assert 0 <= x1 < 155 < x2 <= 300
    assert 0 <= y1 < 155 < y2 <= 300


def test_face_bbox_from_kps_clamps_to_frame_edges_near_a_border() -> None:
    kps = np.array([[5, 5], [15, 5], [10, 10], [6, 15], [14, 15]], dtype=np.float32)
    x1, y1, _x2, _y2 = face_bbox_from_kps((300, 300), kps)
    assert x1 == 0
    assert y1 == 0


def test_detect_face_box_returns_none_without_a_detected_face() -> None:
    detector = _FakeDetector([[]])
    assert detect_face_box(_sharp_image(), detector=detector) is None


def test_detect_face_box_returns_a_box_when_a_face_is_found() -> None:
    detector = _FakeDetector([[_FakeFace(_KPS)]])
    box = detect_face_box(_sharp_image(), detector=detector)
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 < x2 and y1 < y2
