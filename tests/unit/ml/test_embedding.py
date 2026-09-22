"""ml/identity/embedding.py — averaging and the mean self-cosine
measurement. Pure math (no model loading needed) — live-verified
separately against a real set of usable photos (see the module's own
docstring).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.identity.embedding import (
    NoEmbeddingsError,
    average_embedding,
    build_identity_model,
    cosine_similarity,
)
from ml.identity.model import IdentityModel, JsonIdentityModelStore


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def test_average_embedding_is_a_unit_vector() -> None:
    result = average_embedding([_vec(1, 0, 0), _vec(0, 1, 0)])
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_average_embedding_of_identical_vectors_is_that_vector() -> None:
    v = _vec(3, 4, 0)  # not already unit-length, to prove normalization happens
    result = average_embedding([v, v, v])
    assert result == pytest.approx(v / np.linalg.norm(v))


def test_cosine_similarity_identical_vectors_is_one() -> None:
    v = _vec(1, 2, 3)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity(_vec(1, 0), _vec(0, 1)) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one() -> None:
    assert cosine_similarity(_vec(1, 0), _vec(-1, 0)) == pytest.approx(-1.0)


def test_build_identity_model_all_identical_embeddings_gives_perfect_cosine() -> None:
    v = _vec(1, 2, 3, 4)
    model = build_identity_model("me", [v, v, v])
    assert model.person_id == "me"
    assert model.source_photo_count == 3
    assert model.mean_self_cosine == pytest.approx(1.0)


def test_build_identity_model_with_some_variation_scores_below_one() -> None:
    model = build_identity_model("me", [_vec(1, 0, 0), _vec(0.9, 0.1, 0), _vec(0.8, 0.2, 0.1)])
    assert model.mean_self_cosine is not None
    assert 0.0 < model.mean_self_cosine < 1.0


def test_build_identity_model_raises_on_empty_list() -> None:
    with pytest.raises(NoEmbeddingsError):
        build_identity_model("me", [])


def test_json_store_round_trips(tmp_path: Path) -> None:
    store = JsonIdentityModelStore(tmp_path)
    model = build_identity_model("alex", [_vec(1, 0), _vec(0.9, 0.1)])

    store.save(model)
    loaded = store.load("alex")

    assert loaded is not None
    assert loaded == model


def test_json_store_returns_none_for_unknown_person(tmp_path: Path) -> None:
    store = JsonIdentityModelStore(tmp_path)
    assert store.load("nobody") is None


def test_json_store_path_matches_storage_convention(tmp_path: Path) -> None:
    store = JsonIdentityModelStore(tmp_path)
    model = build_identity_model("me", [_vec(1, 0)])
    store.save(model)
    assert (tmp_path / "me" / "identity_model.json").exists()


def test_identity_model_embedding_is_json_serializable_list() -> None:
    model = build_identity_model("me", [_vec(1, 0, 0)])
    assert isinstance(model.embedding, list)
    assert isinstance(model, IdentityModel)
