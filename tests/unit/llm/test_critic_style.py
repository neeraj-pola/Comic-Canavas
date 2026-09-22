"""critic/style.py — thin wrapper over `ml/identity/style_score.py`'s
DINOv2 top-k cosine scoring. `embed_image` is monkeypatched at its own
module (`ml.identity.style_score`), since `critic/style.py` calls the
higher-level `style_score()` function rather than importing
`embed_image` itself — no real model load or network in these tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.critic import style as critic_style
from ml.identity import style_score as style_score_module


def _unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


def test_score_style_uses_the_given_reference_bank(monkeypatch: pytest.MonkeyPatch) -> None:
    query = _unit([1.0, 0.0])
    bank = np.stack([_unit([0.99, 0.14]), _unit([0.0, 1.0])])
    monkeypatch.setattr(critic_style, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(style_score_module, "embed_image", lambda image_bgr: query)

    score = critic_style.score_style(b"fake-jpeg", reference_bank=bank, top_k=1)
    assert score == pytest.approx(float(bank[0] @ query))


def test_score_style_defaults_to_the_committed_reference_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_with = {}

    def fake_load_reference_bank() -> np.ndarray:
        called_with["loaded"] = True
        return np.stack([_unit([1.0, 0.0])])

    monkeypatch.setattr(critic_style, "load_reference_bank", fake_load_reference_bank)
    monkeypatch.setattr(critic_style, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(style_score_module, "embed_image", lambda image_bgr: _unit([1.0, 0.0]))

    critic_style.score_style(b"fake-jpeg")
    assert called_with.get("loaded") is True
