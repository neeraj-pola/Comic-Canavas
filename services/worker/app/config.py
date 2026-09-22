"""Environment configuration for the worker service. Required vars have no
default, so startup fails fast rather than silently running unconfigured.
Kept separate from the API's own `Settings` since the two services'
required config diverges (LLM routing, image generation, etc.)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv
from pydantic import RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATH = find_dotenv(usecwd=True)
_REPO_ROOT = Path(_ENV_PATH).resolve().parent if _ENV_PATH else Path.cwd()


class Settings(BaseSettings):
    # `find_dotenv(usecwd=True)` walks up from the current working directory
    # to find the repo-root `.env` regardless of which directory this
    # process was started from, so it resolves correctly whether run from
    # the repo root or from `services/worker`. Falls back to a bare ".env"
    # when none is found, preserving fail-fast behavior on a fresh clone.
    model_config = SettingsConfigDict(
        env_file=find_dotenv(usecwd=True) or ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- core (required — no default; missing either fails startup) ---
    database_url: str
    redis_url: RedisDsn

    # --- core (safe local defaults) ---
    env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"

    # --- storage (CLAUDE.md §0.2b) ---
    storage: Literal["local", "r2"] = "local"
    local_storage_dir: str = ".data"

    @field_validator("local_storage_dir")
    @classmethod
    def _anchor_to_repo_root(cls, value: str) -> str:
        """Anchors a relative `local_storage_dir` to the repo root rather than
        the current process's cwd, so the API and worker (started from
        different directories) both read/write the same shared directory —
        `LocalFileStorage` is dev's stand-in for a shared R2 bucket. An
        already-absolute value is left untouched."""
        path = Path(value)
        return str(path) if path.is_absolute() else str(_REPO_ROOT / path)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # values come from env/.env
