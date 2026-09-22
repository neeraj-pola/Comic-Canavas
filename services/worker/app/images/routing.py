"""`IMAGE_PROVIDER` routing — resolves which `ImageProvider` to use.

Shaped differently from `llm/routing.py`'s role -> (provider, model) table:
there's one role here, and a provider needs more than a model string to
construct (`MockImageProvider` needs `Storage`, real generators need their
own API key env vars), so this only resolves *which* provider; each factory
reads whatever else it needs itself.

Providers other than mock are constructed on demand (imported inside their
factory functions, not at module load), so resolving "mock" in tests never
requires API keys for the others.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal, cast

from storage import Storage

from app.images.base import ImageProvider
from app.images.mock import MockImageProvider

# Matches `ImagePrompt.generator`'s literal exactly. `nodes/prompts.py`
# imports this as its own `Generator` alias rather than redeclaring the same
# four strings a second place to drift from.
ProviderName = Literal["mock", "leonardo", "flux", "flux_kontext"]


class UnknownImageProviderError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(f"unknown IMAGE_PROVIDER {name!r}; known providers: {sorted(_FACTORIES)}")


def _leonardo_provider(storage: Storage) -> ImageProvider:
    from app.images.leonardo import LeonardoProvider

    return LeonardoProvider(storage)


def _flux_provider(storage: Storage) -> ImageProvider:
    from app.images.fal_flux import FalFluxProvider

    return FalFluxProvider(storage)


def _flux_kontext_provider(storage: Storage) -> ImageProvider:
    from app.images.flux_kontext import FluxKontextProvider

    return FluxKontextProvider(storage)


_FACTORIES: dict[str, Callable[[Storage], ImageProvider]] = {
    "mock": MockImageProvider,
    "leonardo": _leonardo_provider,
    "flux": _flux_provider,
    "flux_kontext": _flux_kontext_provider,
}


def resolve(storage: Storage) -> ImageProvider:
    name = os.environ.get("IMAGE_PROVIDER", "mock")
    if name not in _FACTORIES:
        raise UnknownImageProviderError(name)
    return _FACTORIES[name](storage)


def active_provider_name() -> ProviderName:
    """The real `IMAGE_PROVIDER` as a typed value, for callers (e.g.
    `nodes/prompts.py`) that need to know which dialect to assemble a
    prompt in rather than leaving a `generator` parameter at a stale
    default."""
    name = os.environ.get("IMAGE_PROVIDER", "mock")
    if name not in _FACTORIES:
        raise UnknownImageProviderError(name)
    return cast(ProviderName, name)
