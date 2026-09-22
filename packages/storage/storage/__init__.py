"""Storage interface — `LocalFileStorage` (dev) and `R2Storage` (production)."""

from .base import Storage, StorageError
from .local import LocalFileStorage
from .r2 import R2Storage

__all__ = ["LocalFileStorage", "R2Storage", "Storage", "StorageError"]
