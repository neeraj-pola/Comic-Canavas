"""The image-embedding features of the personal ranking model."""

from __future__ import annotations

import numpy as np

from app.preference import embedding, model
from app.preference.features import (
    EMBED_DIMS,
    FEATURES,
    JUDGE_FEATURES,
    JUDGE_NEUTRAL,
    SCORE_FEATURES,
    feature_vector,
    prior_variances,
)
from app.preference.state import PreferenceState
from contracts import Candidate


def _vectors(n: int, d: int = 32, seed: int = 0) -> list[np.ndarray]:
    return list(np.random.default_rng(seed).normal(size=(n, d)))


def test_there_is_no_projection_until_there_are_enough_embeddings() -> None:
    assert embedding.fit_projection(_vectors(embedding.MIN_VECTORS - 1)) is None
    assert embedding.fit_projection(_vectors(embedding.MIN_VECTORS)) is not None


def test_the_projection_is_deterministic_standardised_and_survives_json() -> None:
    vectors = _vectors(40)
    first = embedding.fit_projection(vectors)
    second = embedding.fit_projection(vectors)
    assert first is not None and second is not None
    assert np.allclose(first.components, second.components)
    projected = np.stack([embedding.project(first, v) for v in vectors])
    assert projected.shape == (40, EMBED_DIMS)
    assert np.allclose(projected.std(axis=0), 1.0, atol=1e-6)
    restored = embedding.Projection.from_json(first.to_json())
    assert restored is not None
    assert np.allclose(embedding.project(restored, vectors[0]), projected[0], atol=1e-3)


def test_no_projection_or_no_embedding_reads_as_all_zero_components() -> None:
    projection = embedding.fit_projection(_vectors(40))
    assert not np.any(embedding.project(None, [0.1] * 32))
    assert not np.any(embedding.project(projection, None))
    assert not np.any(embedding.project(projection, [0.1] * 5))  # wrong size: ignored, not a crash
    assert embedding.Projection.from_json({}) is None


def test_a_candidate_with_no_judge_ratings_reads_as_average_on_them() -> None:
    scores = {f: 0.4 for f in SCORE_FEATURES if f not in JUDGE_FEATURES}
    vector = feature_vector(scores)
    assert vector is not None and len(vector) == len(SCORE_FEATURES)
    judge = vector[[SCORE_FEATURES.index(f) for f in JUDGE_FEATURES]]
    assert list(judge) == [JUDGE_NEUTRAL] * len(JUDGE_FEATURES)
    assert feature_vector({"identity": 0.5}) is None  # still needs the measured signals


def test_embedding_components_get_a_tighter_prior_than_the_named_features() -> None:
    prior = prior_variances()
    assert len(prior) == len(FEATURES)
    assert prior[FEATURES.index("emb_0")] < prior[FEATURES.index("warmth")]


def test_a_tighter_prior_shrinks_a_weight_more() -> None:
    diffs = np.tile(np.array([[1.0, 1.0]]), (6, 1))
    loose, _ = model.fit(diffs, prior_var=np.array([1.0, 1.0]))
    mixed, _ = model.fit(diffs, prior_var=np.array([1.0, 0.05]))
    assert abs(mixed[1]) < abs(loose[1]) and mixed[0] > loose[0]  # the tight one gives way


def test_the_states_vector_appends_the_embedding_components() -> None:
    projection = embedding.fit_projection(_vectors(40))
    state = PreferenceState(
        mu=np.zeros(len(FEATURES)),
        cov=np.eye(len(FEATURES)),
        scales=np.ones(len(FEATURES)),
        n_taps=0,
        active=False,
        projection=projection,
    )
    scores = {f: 0.5 for f in SCORE_FEATURES if f not in JUDGE_FEATURES}
    with_embedding = Candidate(
        id="c", panel_id=1, url="u", seed=1, scores=scores, embedding=list(_vectors(1)[0])
    )
    vector = state.vector(with_embedding)
    assert vector is not None and len(vector) == len(FEATURES)
    assert np.any(vector[len(SCORE_FEATURES) :])
    bare = state.vector(with_embedding.model_copy(update={"embedding": None}))
    assert bare is not None and not np.any(bare[len(SCORE_FEATURES) :])
    assert state.vector(with_embedding.model_copy(update={"scores": {}})) is None
