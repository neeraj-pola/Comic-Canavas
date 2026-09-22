"""Text embeddings for `memory.py`'s pgvector search.

`BAAI/bge-small-en-v1.5`, CPU-only, loaded directly via `transformers`.
384-dim output, CLS-token pooling (per the model card), L2-normalized. BGE's
asymmetric retrieval convention: a query gets a fixed instruction prefix
prepended, indexed passages get none — `embed_query` and `embed_passage` are
separate functions rather than one with a flag, so the two can't be swapped
at a call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model: tuple[PreTrainedTokenizerBase, PreTrainedModel] | None = None


def _get_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    global _model
    if _model is None:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        _model = (tokenizer, model)
    return _model


def _embed(texts: list[str]) -> list[list[float]]:
    import torch

    tokenizer, model = _get_model()
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    cls = output.last_hidden_state[:, 0]
    normalized = torch.nn.functional.normalize(cls, p=2, dim=1)
    result: list[list[float]] = normalized.tolist()
    return result


def embed_passage(text: str) -> list[float]:
    """For text being indexed (a beat's event/place/quote text)."""
    return _embed([text])[0]


def embed_query(text: str) -> list[float]:
    """For a `search()` query — prepends BGE's retrieval instruction."""
    return _embed([QUERY_INSTRUCTION + text])[0]
