"""ml/experiment_tracking.py. When no WANDB_API_KEY is configured (see
the module's own docstring) the real path is mocked here (`wandb.init`),
matching this project's "mock the dependency, not the transport"
convention; the local-fallback path (no key) is real and exercised for
real (real file, real JSON content)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml.experiment_tracking import log_run


@pytest.fixture(autouse=True)
def _no_wandb_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


def test_log_run_falls_back_to_a_local_json_file_without_a_wandb_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ml.experiment_tracking as tracking_module

    monkeypatch.setattr(tracking_module, "_LOCAL_FALLBACK_DIR", tmp_path)

    path_str = log_run(
        kind="reward_model",
        config={"features": ["identity", "style"]},
        snapshot_hash="heldout_week:28pairs",
        metrics={"learned_pairwise_accuracy": 0.43},
        artifact_url="https://huggingface.co/x/reward_head.onnx",
    )

    path = Path(path_str)
    assert path.is_file()
    assert path.parent == tmp_path
    content = path.read_text()
    assert "heldout_week:28pairs" in content
    assert "reward_head.onnx" in content


def test_log_run_calls_wandb_when_a_key_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-key-not-real")

    fake_run = MagicMock()
    fake_run.url = "https://wandb.ai/someuser/comiccanvas/runs/abc123"
    fake_wandb = MagicMock()
    fake_wandb.init.return_value = fake_run
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    url = log_run(
        kind="reward_model",
        config={"features": ["identity"]},
        snapshot_hash="snap-1",
        metrics={"acc": 0.5},
        artifact_url="https://example.com/model.onnx",
    )

    assert url == "https://wandb.ai/someuser/comiccanvas/runs/abc123"
    fake_wandb.init.assert_called_once_with(
        project="comiccanvas", job_type="reward_model", config={"features": ["identity"]}
    )
    fake_run.log.assert_any_call({"acc": 0.5})
    fake_run.finish.assert_called_once()
