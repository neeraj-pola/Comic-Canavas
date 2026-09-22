"""ml/identity/hf_registry.py — mocks `HfApi.upload_file` directly (same
"mock the dependency, not the transport" approach used throughout this
project) rather than a real network call. Live-verified separately: a
real trained LoRA uploaded to the project's actual private HF repo and
confirmed present via the HF API afterward.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ml.identity.hf_registry import MissingHfConfigError, upload_lora


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_REPO", "someuser/comiccanvas-models")
    monkeypatch.setenv("HF_TOKEN", "test-token-not-real")


def test_upload_lora_returns_the_resolve_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_api = MagicMock()
    monkeypatch.setattr("ml.identity.hf_registry.HfApi", lambda token: fake_api)

    url = upload_lora(b"fake-safetensors-bytes", path_in_repo="me/me_lora_v1.safetensors")

    assert url == (
        "https://huggingface.co/someuser/comiccanvas-models/resolve/main/me/me_lora_v1.safetensors"
    )
    fake_api.upload_file.assert_called_once()
    call_kwargs = fake_api.upload_file.call_args.kwargs
    assert call_kwargs["path_in_repo"] == "me/me_lora_v1.safetensors"
    assert call_kwargs["repo_id"] == "someuser/comiccanvas-models"
    assert call_kwargs["repo_type"] == "model"


def test_upload_lora_raises_without_hf_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_REPO", raising=False)
    with pytest.raises(MissingHfConfigError, match="HF_REPO"):
        upload_lora(b"data", path_in_repo="x.safetensors")


def test_upload_lora_raises_without_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(MissingHfConfigError, match="HF_TOKEN"):
        upload_lora(b"data", path_in_repo="x.safetensors")
