"""Exponential backoff on retryable errors, logged, capped attempts."""

from __future__ import annotations

import pytest

from app.llm.base import with_backoff


class FakeRateLimitError(Exception):
    pass


class FakeAuthError(Exception):
    pass


async def test_succeeds_after_two_retryable_failures() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise FakeRateLimitError("429")
        return "ok"

    result = await with_backoff(
        flaky, retryable=(FakeRateLimitError,), max_attempts=3, base_delay_s=0.001
    )
    assert result == "ok"
    assert calls == 3


async def test_gives_up_after_max_attempts() -> None:
    calls = 0

    async def always_fails() -> str:
        nonlocal calls
        calls += 1
        raise FakeRateLimitError("429")

    with pytest.raises(FakeRateLimitError):
        await with_backoff(
            always_fails, retryable=(FakeRateLimitError,), max_attempts=3, base_delay_s=0.001
        )
    assert calls == 3


async def test_non_retryable_error_is_not_retried() -> None:
    calls = 0

    async def bad_request() -> str:
        nonlocal calls
        calls += 1
        raise FakeAuthError("401")

    with pytest.raises(FakeAuthError):
        await with_backoff(
            bad_request, retryable=(FakeRateLimitError,), max_attempts=3, base_delay_s=0.001
        )
    assert calls == 1  # not in the retryable set — fails immediately, no retry
