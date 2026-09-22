"""Pushes a trained LoRA's `.safetensors` bytes to the private HF repo set
up for this project (`HF_REPO`). R2 isn't available yet, so this is HF
alone for now — the same "swap the backing store later" pattern as
`Storage`'s Local/R2 split.
"""

from __future__ import annotations

import io
import os

from huggingface_hub import HfApi


class MissingHfConfigError(ValueError):
    def __init__(self, var: str) -> None:
        super().__init__(f"{var} is not set (see .env.example)")


def upload_lora(data: bytes, *, path_in_repo: str) -> str:
    """Uploads to `HF_REPO`, returns the resulting file's URL."""
    repo_id = os.environ.get("HF_REPO")
    token = os.environ.get("HF_TOKEN")
    if not repo_id:
        raise MissingHfConfigError("HF_REPO")
    if not token:
        raise MissingHfConfigError("HF_TOKEN")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=io.BytesIO(data),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )
    return f"https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}"
