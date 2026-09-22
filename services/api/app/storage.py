"""`Storage` factory — builds the adapter from `Settings`, mirroring how
`IMAGE_PROVIDER`/`LLM_*` route by env var elsewhere in this project. R2
isn't wired here yet — `Settings` has no R2 fields until real credentials
exist for it.
"""

from __future__ import annotations

from functools import lru_cache

from storage import LocalFileStorage, Storage

from app.config import get_settings


@lru_cache
def get_storage() -> Storage:
    settings = get_settings()
    if settings.storage == "r2":
        raise NotImplementedError("R2Storage wiring is not built yet")
    return LocalFileStorage(
        root=settings.local_storage_dir,
        base_url=settings.api_base_url,
        secret=settings.storage_secret,
    )
