"""Environment configuration for the scheduler service."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from dotenv import find_dotenv
from pydantic import RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # A relative `env_file=".env"` only resolves when cwd is the repo root;
    # `find_dotenv(usecwd=True)` finds it regardless of cwd.
    model_config = SettingsConfigDict(
        env_file=find_dotenv(usecwd=True) or ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    redis_url: RedisDsn

    env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # values come from env/.env
