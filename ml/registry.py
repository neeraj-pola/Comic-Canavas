"""Every trained artifact goes to the HF Hub private repo (`HF_REPO`) —
`ml/identity/hf_registry.py`'s `upload_lora` is reused here, not
reimplemented, since uploading arbitrary bytes to a path in that repo
isn't actually LoRA-specific despite the name — with tags
`kind`/`data_snapshot`/`metrics`. The real `checkpoints` table mirrors it,
which is what `list_checkpoints` actually reads from (the DB row, not a
live HF API call), so it stays fast and works even if HF is briefly
unreachable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import find_dotenv, load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Json

from ml.identity.hf_registry import upload_lora as upload_artifact

__all__ = ["list_checkpoints", "register_checkpoint", "upload_artifact"]


def _connect(**kwargs: Any) -> psycopg.Connection[Any]:
    load_dotenv(find_dotenv(usecwd=True))
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (see .env.example)")
    conn = psycopg.connect(url, **kwargs)
    schema = os.environ.get("DB_SCHEMA")
    if schema:
        conn.execute(f'SET search_path TO "{schema}", public')
    return conn


def register_checkpoint(
    *,
    checkpoint_id: str,
    kind: str,
    artifact_path: Path,
    path_in_repo: str,
    data_snapshot: str,
    metrics: dict[str, Any],
) -> str:
    """Uploads `artifact_path`'s real bytes to the HF Hub, then mirrors it
    into the `checkpoints` table. Returns the real HF resolve URL.

    `load_dotenv` runs here, before `upload_artifact` — that function
    (`upload_lora`) reads `HF_REPO`/`HF_TOKEN` from `os.environ` directly
    and doesn't load `.env` itself, so calling it first would raise
    `MissingHfConfigError` even with a real `.env` on disk."""
    load_dotenv(find_dotenv(usecwd=True))
    artifact_url = upload_artifact(artifact_path.read_bytes(), path_in_repo=path_in_repo)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO checkpoints (id, kind, data_snapshot, metrics, artifact_url)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                data_snapshot = EXCLUDED.data_snapshot,
                metrics = EXCLUDED.metrics,
                artifact_url = EXCLUDED.artifact_url
            """,
            (checkpoint_id, kind, data_snapshot, Json(metrics), artifact_url),
        )
        conn.commit()
    return artifact_url


def list_checkpoints(kind: str) -> list[dict[str, Any]]:
    with _connect(row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT id, kind, data_snapshot, metrics, artifact_url, created_at "
            "FROM checkpoints WHERE kind = %s ORDER BY created_at DESC",
            (kind,),
        ).fetchall()
    return [dict(r) for r in rows]
