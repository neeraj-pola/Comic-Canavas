"""Unauthenticated request -> 401; valid (mock) token -> user row
upserted."""

from __future__ import annotations

from collections.abc import Callable

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow

pytestmark = pytest.mark.enable_socket


def test_missing_bearer_token_is_401(client: TestClient) -> None:
    response = client.get("/settings")
    assert response.status_code == 401


def test_valid_mock_token_upserts_the_user_row(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    response = client.get("/settings", headers=auth_headers("auth-test-user"))
    assert response.status_code == 200

    row = conn.execute("SELECT id FROM users WHERE id = %s", ("auth-test-user",)).fetchone()
    assert row is not None
