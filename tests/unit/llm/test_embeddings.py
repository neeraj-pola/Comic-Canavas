"""app/embeddings.py — BGE-small CLS-pooled, normalized embeddings.
`_get_model` is monkeypatched with fake tokenizer/model objects; these
tests never download or run the real ~130MB model.
"""

from __future__ import annotations

import pytest
import torch

from app import embeddings


class _FakeTokenizer:
    def __init__(self) -> None:
        self.last_call: dict[str, object] = {}

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        self.last_call = {
            "texts": texts,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "return_tensors": return_tensors,
        }
        return {"input_ids": torch.zeros((len(texts), 3), dtype=torch.long)}


class _FakeOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeModel:
    """`cls_vectors` is what the real model would put at each sequence's
    first ([CLS]) position; other positions are zeroed so a test that
    accidentally mean-pools instead of CLS-pooling fails visibly."""

    def __init__(self, cls_vectors: torch.Tensor) -> None:
        self._cls_vectors = cls_vectors

    def eval(self) -> None:
        pass

    def __call__(self, **kwargs: object) -> _FakeOutput:
        batch, dim = self._cls_vectors.shape
        hidden = torch.zeros((batch, 3, dim))
        hidden[:, 0, :] = self._cls_vectors
        return _FakeOutput(hidden)


def test_embed_passage_uses_the_cls_token_and_l2_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cls = torch.tensor([[3.0, 4.0]])  # norm 5 -> normalized (0.6, 0.8)
    monkeypatch.setattr(embeddings, "_get_model", lambda: (_FakeTokenizer(), _FakeModel(cls)))

    result = embeddings.embed_passage("some text")

    assert result == pytest.approx([0.6, 0.8])


def test_embed_query_prepends_the_retrieval_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    """BGE's asymmetric retrieval convention means a query without the
    instruction prefix silently ranks worse, not an error — so the exact
    prefixed text sent to the tokenizer must be verified, not just that
    *some* embedding comes back."""
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        embeddings, "_get_model", lambda: (tokenizer, _FakeModel(torch.tensor([[1.0, 0.0]])))
    )

    embeddings.embed_query("gym")

    assert tokenizer.last_call["texts"] == [embeddings.QUERY_INSTRUCTION + "gym"]


def test_embed_passage_does_not_prepend_the_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        embeddings, "_get_model", lambda: (tokenizer, _FakeModel(torch.tensor([[1.0, 0.0]])))
    )

    embeddings.embed_passage("gym")

    assert tokenizer.last_call["texts"] == ["gym"]


def test_embed_uses_max_length_512_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        embeddings, "_get_model", lambda: (tokenizer, _FakeModel(torch.tensor([[1.0, 0.0]])))
    )

    embeddings.embed_passage("some text")

    assert tokenizer.last_call["max_length"] == 512
    assert tokenizer.last_call["truncation"] is True
