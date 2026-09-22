"""Environment configuration for the API service. Required vars have no
default, so startup fails fast rather than limping along against an
unconfigured database. See `.env.example` for the full annotated list —
this class only pins the subset the API needs to boot.
"""

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
    # A relative `env_file=".env"` only resolves when cwd is the repo root;
    # `find_dotenv(usecwd=True)` finds it regardless of cwd.
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

    # --- auth ---
    auth_mode: Literal["mock", "clerk"] = "mock"
    clerk_secret_key: str | None = None
    clerk_jwks_url: str | None = None

    # --- storage ---
    storage: Literal["local", "r2"] = "local"
    local_storage_dir: str = ".data"
    api_base_url: str = "http://localhost:8000"
    storage_secret: str = "dev-local-storage-secret"

    # --- CORS (apps/web is a different origin) ---
    # A regex, not a fixed port list: `next dev` picks the next free port
    # when 3000 is taken, and Playwright's own dev server uses yet another
    # one — a fixed allow-list would need updating every time. Locked to a
    # real origin in production; this default only matches
    # localhost/127.0.0.1 on any port.
    cors_allow_origin_regex: str = r"http://(localhost|127\.0\.0\.1):\d+"

    @field_validator("local_storage_dir")
    @classmethod
    def _anchor_to_repo_root(cls, value: str) -> str:
        """Anchors a relative `local_storage_dir` to the repo root rather
        than the current process's cwd, so the API and worker (started
        from different directories) read/write the same shared directory."""
        path = Path(value)
        return str(path) if path.is_absolute() else str(_REPO_ROOT / path)


@lru_cache
def get_settings() -> Settings:
    """Cached so `.env` is parsed once per process, not per request."""
    return Settings()  # values come from env/.env
