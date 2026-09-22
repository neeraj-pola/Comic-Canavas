"""Every training run logs config, snapshot hash, metrics, and artifact
URL to W&B, project `comiccanvas`.

Without a real `WANDB_API_KEY`, `log_run` falls back to writing the same
payload to a local JSON file under `ml/experiment_runs/` — a real,
inspectable record, not a silent no-op — so callers always have somewhere
real to look regardless of whether W&B is configured.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WANDB_PROJECT = "comiccanvas"
_LOCAL_FALLBACK_DIR = Path(__file__).resolve().parent / "experiment_runs"


def log_run(
    *,
    kind: str,
    config: dict[str, Any],
    snapshot_hash: str,
    metrics: dict[str, Any],
    artifact_url: str | None = None,
) -> str:
    """Returns the real W&B run URL when `WANDB_API_KEY` is set, else the
    local fallback file's path — callers that want to show "where did this
    run go" always get a real, dereferenceable answer."""
    api_key = os.environ.get("WANDB_API_KEY")
    payload = {
        "kind": kind,
        "config": config,
        "snapshot_hash": snapshot_hash,
        "metrics": metrics,
        "artifact_url": artifact_url,
        "logged_at": datetime.now(UTC).isoformat(),
    }

    if not api_key:
        _LOCAL_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOCAL_FALLBACK_DIR / f"{kind}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.json"
        path.write_text(json.dumps(payload, indent=2))
        return str(path)

    import wandb

    run = wandb.init(project=WANDB_PROJECT, job_type=kind, config=config)
    try:
        run.log(metrics)
        if artifact_url:
            run.log({"artifact_url": artifact_url})
        run.summary["snapshot_hash"] = snapshot_hash
        url = run.url or ""
    finally:
        run.finish()
    return url
