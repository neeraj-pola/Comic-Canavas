"""IMAGE_PROVIDER routing."""

from __future__ import annotations

from pathlib import Path

import pytest
from storage import LocalFileStorage

from app.images import routing
from app.images.mock import MockImageProvider


def _storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")


def test_defaults_to_mock_when_env_var_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IMAGE_PROVIDER", raising=False)
    provider = routing.resolve(_storage(tmp_path))
    assert isinstance(provider, MockImageProvider)


def test_resolves_mock_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_PROVIDER", "mock")
    provider = routing.resolve(_storage(tmp_path))
    assert isinstance(provider, MockImageProvider)


@pytest.mark.parametrize(
    ("name", "missing_var"),
    [("leonardo", "LEONARDO_API_KEY"), ("flux", "FAL_KEY")],
)
def test_real_providers_fail_clearly_without_api_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, missing_var: str
) -> None:
    # Neither has live credentials in this environment (tasks 4.3/4.4 are
    # unverified — see their modules' docstrings) — constructing either
    # should fail with a clear "which var is missing" error, not a
    # confusing one several calls deep.
    monkeypatch.setenv("IMAGE_PROVIDER", name)
    monkeypatch.delenv("LEONARDO_API_KEY", raising=False)
    monkeypatch.delenv("LEONARDO_MODEL_ID", raising=False)
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(Exception, match=missing_var):
        routing.resolve(_storage(tmp_path))


def test_unknown_provider_raises_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGE_PROVIDER", "midjourney")
    with pytest.raises(routing.UnknownImageProviderError, match="midjourney"):
        routing.resolve(_storage(tmp_path))
