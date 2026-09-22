"""CORS middleware config for cross-origin calls from apps/web. `TestClient`
runs the real ASGI middleware stack, so this exercises the actual
`CORSMiddleware` config, not a mock."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.enable_socket


def test_localhost_dev_origin_is_allowed(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    headers = {**auth_headers("cors-test-user"), "Origin": "http://localhost:3001"}
    response = client.get("/settings", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3001"


def test_an_unrelated_origin_is_not_allowed(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    headers = {**auth_headers("cors-test-user-2"), "Origin": "https://evil.example"}
    response = client.get("/settings", headers=headers)
    # the request itself still succeeds (CORS is enforced by the browser,
    # not the server) — what matters is the response carries no
    # allow-origin header for a browser to actually honor
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_preflight_request_is_answered_for_an_allowed_origin(client: TestClient) -> None:
    response = client.options(
        "/settings",
        headers={
            "Origin": "http://localhost:3100",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3100"
