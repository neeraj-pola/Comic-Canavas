"""`GET/PUT /settings` — round-trip test."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.enable_socket


def test_get_settings_returns_defaults_when_none_saved(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    response = client.get("/settings", headers=auth_headers("settings-user"))
    assert response.status_code == 200
    assert response.json()["humor"] == 5
    assert response.json()["generator"] == "mock"


def test_put_then_get_round_trips(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    headers = auth_headers("settings-user-2")
    body = {
        "style": "warm",
        "humor": 8,
        "caption_length": "short",
        "sensitive_mode": False,
        "reminder_time": "21:00",
        "channel": "email",
        "generator": "flux",
        "detail_level": "high",
        "personalise": False,
    }

    put_response = client.put("/settings", json=body, headers=headers)
    assert put_response.status_code == 200

    get_response = client.get("/settings", headers=headers)
    assert get_response.status_code == 200
    assert get_response.json() == body
