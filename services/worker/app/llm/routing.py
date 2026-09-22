"""Role -> (provider, model) routing table.

One env var flips a role between providers: `LLM_EXTRACTOR=openai:gpt-4o-mini`
vs `LLM_EXTRACTOR=anthropic:claude-sonnet-4-6`. `validate_routing()` runs at
worker startup so a typo'd provider name fails immediately and loudly
instead of surfacing mid-pipeline the first time some node calls `resolve()`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable

from app.llm.base import LLMProvider
from app.llm.mock_provider import MockProvider

ROLES: tuple[str, ...] = ("extractor", "script", "prompts", "judge", "lookcard", "critic")

# Roles that work without their env var, so adding one never breaks an existing `.env`.
# `critic` is the per-panel vision judge (`critic/judge.py`): a small, cheap vision model,
# called once per panel with the reference and all three candidates (~$0.002 a panel).
DEFAULT_SPECS: dict[str, str] = {"critic": "anthropic:claude-haiku-4-5-20251001"}


class MissingRoutingError(ValueError):
    def __init__(self, env_var: str) -> None:
        super().__init__(f"{env_var} is not set (see .env.example)")


class UnknownProviderError(ValueError):
    def __init__(self, provider_name: str, spec: str) -> None:
        super().__init__(
            f"unknown LLM provider {provider_name!r} (from {spec!r}); "
            f"known providers: {sorted(_PROVIDER_FACTORIES)}"
        )


class MalformedRoutingSpecError(ValueError):
    def __init__(self, spec: str) -> None:
        super().__init__(f"malformed LLM routing spec {spec!r}, expected 'provider:model'")


def _anthropic_provider() -> LLMProvider:
    from app.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider()


def _openai_provider() -> LLMProvider:
    from app.llm.openai_provider import OpenAIProvider

    return OpenAIProvider()


# Deferred imports (above) so resolving "mock" in tests never requires the
# `anthropic`/`openai` SDKs to be importable or their API keys to be set.
_PROVIDER_FACTORIES: dict[str, Callable[[], LLMProvider]] = {
    "anthropic": _anthropic_provider,
    "openai": _openai_provider,
    "mock": MockProvider,
}

_provider_cache: dict[str, LLMProvider] = {}


def reset_provider_cache() -> None:
    """Test-only: forget cached provider instances (e.g. after changing an API key env var)."""
    _provider_cache.clear()


def _env_var(role: str) -> str:
    return f"LLM_{role.upper()}"


def parse_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise MalformedRoutingSpecError(spec)
    provider_name, model = spec.split(":", 1)
    if provider_name not in _PROVIDER_FACTORIES:
        raise UnknownProviderError(provider_name, spec)
    return provider_name, model


def _spec(role: str) -> str | None:
    return os.environ.get(_env_var(role)) or DEFAULT_SPECS.get(role)


def validate_routing(roles: Iterable[str] = ROLES) -> dict[str, tuple[str, str]]:
    """Parse and validate every role's env var. Call at process startup."""
    resolved = {}
    for role in roles:
        env_var = _env_var(role)
        spec = _spec(role)
        if not spec:
            raise MissingRoutingError(env_var)
        resolved[role] = parse_spec(spec)
    return resolved


def resolve(role: str) -> tuple[LLMProvider, str]:
    env_var = _env_var(role)
    spec = _spec(role)
    if not spec:
        raise MissingRoutingError(env_var)
    provider_name, model = parse_spec(spec)
    if provider_name not in _provider_cache:
        _provider_cache[provider_name] = _PROVIDER_FACTORIES[provider_name]()
    return _provider_cache[provider_name], model
