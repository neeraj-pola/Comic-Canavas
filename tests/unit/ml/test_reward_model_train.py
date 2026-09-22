"""ml/reward_model/train.py's pure functions — no real DB or HF calls
needed (`load_pairs`/`_insert_checkpoint_and_run` are the only IO,
exercised separately by `main()`'s own live run). Synthetic feature
vectors here are deliberately NOT the real mock-image data (which has
near-zero variance on identity/detail, see the module's own docstring)
— they're constructed so the "correct" direction is obvious, to test
the training/scoring mechanism itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from ml.reward_model.train import (
    FEATURES,
    HAND_SET_WEIGHTS,
    PreferencePair,
    export_onnx,
    pairwise_accuracy,
    split_train_eval,
    train_reward_head,
)


def _pair(chosen: list[float], rejected: list[float]) -> PreferencePair:
    return PreferencePair(
        chosen=np.array(chosen, dtype=np.float32), rejected=np.array(rejected, dtype=np.float32)
    )


# Style is deliberately the ONLY feature that differs, and chosen always
# has the higher one — an unambiguous, learnable synthetic signal.
CLEAR_PAIRS = [
    _pair([0.5, 0.9, 0.5, 0.5], [0.5, 0.1, 0.5, 0.5]),
    _pair([0.3, 0.8, 0.4, 0.6], [0.3, 0.2, 0.4, 0.6]),
    _pair([0.7, 0.7, 0.3, 0.5], [0.7, 0.3, 0.3, 0.5]),
    _pair([0.2, 0.6, 0.6, 0.4], [0.2, 0.4, 0.6, 0.4]),
    _pair([0.6, 0.9, 0.2, 0.7], [0.6, 0.1, 0.2, 0.7]),
    _pair([0.4, 0.8, 0.5, 0.3], [0.4, 0.2, 0.5, 0.3]),
    _pair([0.5, 0.7, 0.7, 0.5], [0.5, 0.3, 0.7, 0.5]),
    _pair([0.3, 0.6, 0.4, 0.6], [0.3, 0.4, 0.4, 0.6]),
]


def test_features_has_the_four_real_critic_signals() -> None:
    assert FEATURES == ("identity", "style", "alignment", "detail")


def test_pairwise_accuracy_is_1_when_weights_agree_with_every_pair() -> None:
    # Weight only on style (index 1), which is what CLEAR_PAIRS varies.
    weights = {"identity": 0.0, "style": 1.0, "alignment": 0.0, "detail": 0.0}
    assert pairwise_accuracy(weights, CLEAR_PAIRS) == 1.0


def test_pairwise_accuracy_is_0_when_weights_are_inverted() -> None:
    weights = {"identity": 0.0, "style": -1.0, "alignment": 0.0, "detail": 0.0}
    assert pairwise_accuracy(weights, CLEAR_PAIRS) == 0.0


def test_hand_set_weights_matches_the_real_critic_reward_head() -> None:
    # Mirrors services/worker/app/critic/reward_head.py's real WEIGHTS —
    # this test is the guard against the two silently drifting apart.
    assert HAND_SET_WEIGHTS == {
        "identity": 0.35,
        "style": 0.25,
        "alignment": 0.25,
        "detail": 0.15,
    }
    assert sum(HAND_SET_WEIGHTS.values()) == pytest.approx(1.0)


def test_split_train_eval_sizes_and_determinism() -> None:
    train, eval_ = split_train_eval(CLEAR_PAIRS, eval_fraction=0.25, seed=1)
    assert len(train) == 6
    assert len(eval_) == 2

    train2, eval2 = split_train_eval(CLEAR_PAIRS, eval_fraction=0.25, seed=1)
    assert [id(p) for p in train] == [id(p) for p in train2]
    assert [id(p) for p in eval_] == [id(p) for p in eval2]


def test_train_reward_head_learns_the_obvious_synthetic_signal() -> None:
    model = train_reward_head(CLEAR_PAIRS, epochs=300, lr=0.1)
    weights = model.weights()

    # Style should dominate — it's the only feature that ever differs.
    assert weights["style"] > 0
    assert pairwise_accuracy(weights, CLEAR_PAIRS) >= 0.75


def test_export_onnx_produces_a_model_matching_the_torch_output(tmp_path: Path) -> None:
    import torch

    model = train_reward_head(CLEAR_PAIRS, epochs=50, lr=0.1)
    onnx_path = tmp_path / "reward_head.onnx"
    export_onnx(model, onnx_path)

    session = ort.InferenceSession(str(onnx_path))
    sample = np.array([[0.5, 0.9, 0.5, 0.5]], dtype=np.float32)

    onnx_output = session.run(None, {"features": sample})[0]
    torch_output = model(torch.tensor(sample)).detach().numpy()

    assert onnx_output == pytest.approx(torch_output, abs=1e-5)
