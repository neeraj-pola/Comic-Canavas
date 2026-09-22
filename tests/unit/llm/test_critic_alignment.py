"""critic/alignment.py — SigLIP text-image similarity. `_get_model` is
monkeypatched to return fake processor/model objects; these tests never
load the real ~380MB SigLIP model or touch the network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.critic import alignment as critic_alignment


class _FakeOutputs:
    def __init__(self, logit: float) -> None:
        self.logits_per_image = torch.tensor([[logit]])


class _FakeModel:
    def __init__(self, logit: float) -> None:
        self._logit = logit
        self.calls: list[dict[str, object]] = []

    def __call__(self, **inputs: object) -> _FakeOutputs:
        self.calls.append(inputs)
        return _FakeOutputs(self._logit)

    def eval(self) -> None:
        pass


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self, *, text: object, images: object, padding: str, return_tensors: str
    ) -> dict[str, object]:
        call = {
            "text": text,
            "images": images,
            "padding": padding,
            "return_tensors": return_tensors,
        }
        self.calls.append(call)
        return call


def test_score_alignment_returns_sigmoid_of_the_logit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _FakeModel(logit=2.0)
    fake_processor = _FakeProcessor()
    monkeypatch.setattr(critic_alignment, "_get_model", lambda: (fake_processor, fake_model))

    score = critic_alignment.score_alignment(
        np.zeros((10, 10, 3), dtype=np.uint8), "a person cooking"
    )

    assert score == pytest.approx(float(torch.sigmoid(torch.tensor(2.0))))


def test_score_alignment_requires_max_length_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    """SigLIP was trained with `padding="max_length"` — omitting it
    silently changes the score rather than raising an error."""
    fake_model = _FakeModel(logit=0.0)
    fake_processor = _FakeProcessor()
    monkeypatch.setattr(critic_alignment, "_get_model", lambda: (fake_processor, fake_model))

    critic_alignment.score_alignment(np.zeros((10, 10, 3), dtype=np.uint8), "a caption")

    assert fake_processor.calls[0]["padding"] == "max_length"


def test_score_alignment_passes_text_as_a_single_element_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _FakeModel(logit=0.0)
    fake_processor = _FakeProcessor()
    monkeypatch.setattr(critic_alignment, "_get_model", lambda: (fake_processor, fake_model))

    critic_alignment.score_alignment(np.zeros((10, 10, 3), dtype=np.uint8), "cooking dinner")

    assert fake_processor.calls[0]["text"] == ["cooking dinner"]


def test_score_alignment_is_higher_for_a_plausible_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not a real model call (mocked), but proves the score direction:
    a higher SigLIP logit for the correct caption must produce a higher
    final score than a lower logit for a mismatched one."""

    def _model_for(logit: float) -> _FakeModel:
        return _FakeModel(logit=logit)

    monkeypatch.setattr(critic_alignment, "_get_model", lambda: (_FakeProcessor(), _model_for(3.0)))
    correct_score = critic_alignment.score_alignment(
        np.zeros((10, 10, 3), dtype=np.uint8), "cooking dinner at the stove"
    )
    monkeypatch.setattr(
        critic_alignment, "_get_model", lambda: (_FakeProcessor(), _model_for(-2.0))
    )
    shuffled_score = critic_alignment.score_alignment(
        np.zeros((10, 10, 3), dtype=np.uint8), "reading a book in bed"
    )
    assert correct_score > shuffled_score
