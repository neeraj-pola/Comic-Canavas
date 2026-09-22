"""ml/identity/style_score.py's pure-math layer. `embed_image` (the real
DINOv2 model call) is monkeypatched throughout — these tests never load
a model or touch the network; the API itself was verified live
separately (see the module's own docstring).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.identity import style_score


def _unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


def test_cosine_to_is_one_for_a_row_already_in_the_bank() -> None:
    bank = np.stack([_unit([1.0, 0.0, 0.0]), _unit([0.0, 1.0, 0.0])])
    cosines = style_score.cosine_to(bank[0], bank)
    assert cosines[0] == pytest.approx(1.0)


def test_cosine_to_is_order_invariant() -> None:
    bank = np.stack([_unit([1.0, 0.0]), _unit([0.0, 1.0]), _unit([0.7, 0.7])])
    query = _unit([1.0, 0.1])
    forward = style_score.cosine_to(query, bank)
    backward = style_score.cosine_to(query, bank[::-1])
    assert sorted(forward) == pytest.approx(sorted(backward))


def test_style_score_uses_top_k_mean_not_all_n_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bank rigged so top-2 mean and all-5 mean genuinely differ: two rows
    # near-identical to the query, three rows far from it.
    query = _unit([1.0, 0.0])
    near = _unit([0.99, 0.14])
    far = _unit([0.0, 1.0])
    bank = np.stack([near, near, far, far, far])

    monkeypatch.setattr(style_score, "embed_image", lambda image_bgr: query)

    top2 = style_score.style_score(np.zeros((4, 4, 3), dtype=np.uint8), bank, top_k=2)
    all5 = style_score.style_score(np.zeros((4, 4, 3), dtype=np.uint8), bank, top_k=5)
    assert top2 > all5


def test_build_and_load_reference_bank_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "comic_character"
    image_dir.mkdir()
    for i in range(3):
        (image_dir / f"{i:03d}.jpg").write_bytes(b"fake-jpeg-bytes")

    embeddings = iter([_unit([1.0, 0.0, 0.0]), _unit([0.0, 1.0, 0.0]), _unit([0.0, 0.0, 1.0])])
    monkeypatch.setattr(style_score, "embed_jpeg", lambda data: next(embeddings))

    out_path = tmp_path / "bank.npy"
    built = style_score.build_reference_bank(image_dir, out_path)
    assert built.shape == (3, 3)

    loaded = style_score.load_reference_bank(out_path)
    np.testing.assert_array_equal(loaded, built)
    # every row should already be unit-normalized
    norms = np.linalg.norm(loaded, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_build_reference_bank_raises_on_empty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(style_score.EmptyReferenceSetError):
        style_score.build_reference_bank(empty_dir, tmp_path / "bank.npy")


class _FakeDetected:
    def __init__(self, crop_bgr: np.ndarray) -> None:
        self.crop_bgr = crop_bgr


def test_master_face_embedding_returns_none_without_a_detected_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(style_score, "detect_and_embed", lambda image_bgr: None)
    assert style_score.master_face_embedding(np.zeros((4, 4, 3), dtype=np.uint8)) is None


def test_master_face_embedding_embeds_the_face_crop_not_the_whole_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop = np.full((2, 2, 3), 7, dtype=np.uint8)
    monkeypatch.setattr(style_score, "detect_and_embed", lambda image_bgr: _FakeDetected(crop))
    seen = {}

    def _fake_embed(image_bgr: np.ndarray) -> np.ndarray:
        seen["image_bgr"] = image_bgr
        return _unit([1.0, 0.0])

    monkeypatch.setattr(style_score, "embed_image", _fake_embed)
    style_score.master_face_embedding(np.zeros((10, 10, 3), dtype=np.uint8))
    np.testing.assert_array_equal(seen["image_bgr"], crop)


def test_identity_similarity_returns_none_without_a_detected_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(style_score, "detect_and_embed", lambda image_bgr: None)
    result = style_score.identity_similarity(np.zeros((4, 4, 3), dtype=np.uint8), _unit([1.0, 0.0]))
    assert result is None


def test_identity_similarity_compares_face_crops_not_whole_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparing whole frames underscores visually on-identity panels;
    the comparison must be crop-to-crop, not frame-to-frame."""
    monkeypatch.setattr(
        style_score, "detect_and_embed", lambda image_bgr: _FakeDetected(np.zeros((2, 2, 3)))
    )
    # Same vector for both sides — a genuinely identical face crop should
    # score a perfect 1.0, proving the comparison is crop-to-crop.
    monkeypatch.setattr(style_score, "embed_image", lambda image_bgr: _unit([1.0, 0.0, 0.0]))

    master_embedding = _unit([1.0, 0.0, 0.0])
    result = style_score.identity_similarity(
        np.zeros((10, 10, 3), dtype=np.uint8), master_embedding
    )
    assert result == pytest.approx(1.0)


def test_within_sheet_similarity_is_one_for_identical_embeddings() -> None:
    embeddings = [_unit([1.0, 0.0]) for _ in range(4)]
    assert style_score.within_sheet_similarity(embeddings) == pytest.approx(1.0)


def test_within_sheet_similarity_is_one_for_fewer_than_two_embeddings() -> None:
    assert style_score.within_sheet_similarity([]) == 1.0
    assert style_score.within_sheet_similarity([_unit([1.0, 0.0])]) == 1.0


def test_within_sheet_similarity_reflects_real_spread() -> None:
    tight = [_unit([1.0, 0.0]), _unit([0.99, 0.14]), _unit([0.98, 0.2])]
    spread = [_unit([1.0, 0.0]), _unit([0.0, 1.0]), _unit([-1.0, 0.0])]
    assert style_score.within_sheet_similarity(tight) > style_score.within_sheet_similarity(spread)
