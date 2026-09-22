"""role -> (provider, model) routing.

Each role must resolve, and an unknown provider must error at startup,
not at call time. `validate_routing()` is the
startup check (would run in `app/worker.py`'s `on_startup`, alongside the
existing healthcheck task) — it parses every role's spec eagerly, without
constructing or calling any actual provider, so a typo'd provider name
fails before a single LLM call is ever attempted.
"""

from __future__ import annotations

import pytest

from app.llm import routing
from app.llm.mock_provider import MockProvider


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    routing.reset_provider_cache()


def _set_all_roles(monkeypatch: pytest.MonkeyPatch, spec: str) -> None:
    for role in routing.ROLES:
        monkeypatch.setenv(f"LLM_{role.upper()}", spec)


def test_each_role_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_all_roles(monkeypatch, "mock:mock-model")
    for role in routing.ROLES:
        provider, model = routing.resolve(role)
        assert isinstance(provider, MockProvider)
        assert model == "mock-model"


def test_validate_routing_returns_every_role(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_all_roles(monkeypatch, "mock:mock-model")
    resolved = routing.validate_routing()
    assert resolved == {role: ("mock", "mock-model") for role in routing.ROLES}


def test_unknown_provider_caught_by_startup_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_all_roles(monkeypatch, "mock:mock-model")
    monkeypatch.setenv("LLM_JUDGE", "not_a_real_provider:some-model")

    # The failure is caught by validate_routing() alone — no provider was
    # constructed and no call was made to reach this error.
    with pytest.raises(routing.UnknownProviderError, match="not_a_real_provider"):
        routing.validate_routing()


def test_unknown_provider_also_errors_on_direct_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_JUDGE", "not_a_real_provider:some-model")
    with pytest.raises(routing.UnknownProviderError):
        routing.resolve("judge")


def test_missing_env_var_errors_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_EXTRACTOR", raising=False)
    with pytest.raises(routing.MissingRoutingError, match="LLM_EXTRACTOR"):
        routing.resolve("extractor")


def test_resolving_mock_never_requires_real_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # No ANTHROPIC_API_KEY / OPENAI_API_KEY set — resolving "mock" must not
    # even import the anthropic/openai SDKs, let alone need credentials.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_EXTRACTOR", "mock:mock-model")
    provider, model = routing.resolve("extractor")
    assert isinstance(provider, MockProvider)
    assert model == "mock-model"


def test_malformed_spec_errors_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EXTRACTOR", "no-colon-here")
    with pytest.raises(routing.MalformedRoutingSpecError):
        routing.resolve("extractor")
