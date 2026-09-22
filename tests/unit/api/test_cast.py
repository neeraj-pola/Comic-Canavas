"""Cast CRUD, photo presign, and the process/train guard rails.

Real face detection (`ml.identity.faces.process_batch`, buffalo_l) is
already covered by `tests/unit/ml/test_faces.py` (a fake detector
injected there to avoid a ~280MB real model load per test) — this file
tests the API's wiring and boundary cases (no photos yet, confirm=false),
not face-detection accuracy again.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow

pytestmark = pytest.mark.enable_socket


def test_create_and_list_cast(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    headers = auth_headers("cast-user")

    create = client.post("/cast", json={"name": "Sam"}, headers=headers)
    assert create.status_code == 201
    person_id = create.json()["id"]

    listing = client.get("/cast", headers=headers)
    assert listing.status_code == 200
    people = listing.json()["people"]
    assert len(people) == 1
    assert people[0]["id"] == person_id
    assert people[0]["name"] == "Sam"
    assert people[0]["identity"] is None


def test_create_person_with_gender_term_and_patch_it_later(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    """`gender_term` is a user-provided value (never vision-inferred). `POST /cast` accepts it up
    front; `PATCH /cast/{id}` covers the case where a person is auto-created (just a name) before
    there's been any chance to ask."""
    headers = auth_headers("cast-user-gender")

    create = client.post("/cast", json={"name": "Sam", "gender_term": "woman"}, headers=headers)
    assert create.status_code == 201
    assert create.json()["gender_term"] == "woman"
    person_id = create.json()["id"]

    listing = client.get("/cast", headers=headers)
    assert listing.json()["people"][0]["gender_term"] == "woman"

    patch = client.patch(f"/cast/{person_id}", json={"gender_term": "man"}, headers=headers)
    assert patch.status_code == 200
    assert patch.json()["gender_term"] == "man"

    listing_after = client.get("/cast", headers=headers)
    assert listing_after.json()["people"][0]["gender_term"] == "man"


def test_photo_presign_writes_a_pending_person_photos_row(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    headers = auth_headers("cast-user-2")
    person_id = client.post("/cast", json={"name": "Sam"}, headers=headers).json()["id"]

    response = client.post(
        f"/cast/{person_id}/photos/presign",
        json={"filename": "photo1.jpg"},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["key"] == f"people/{person_id}/raw/photo1.jpg"
    row = conn.execute(
        "SELECT status FROM person_photos WHERE person_id = %s", (person_id,)
    ).fetchone()
    assert row is not None
    assert row["status"] == "pending"


def test_process_422s_with_no_photos_uploaded(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    headers = auth_headers("cast-user-3")
    person_id = client.post("/cast", json={"name": "Sam"}, headers=headers).json()["id"]

    response = client.post(f"/cast/{person_id}/process", headers=headers)

    assert response.status_code == 422


def test_train_422s_without_confirm(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    headers = auth_headers("cast-user-4")
    person_id = client.post("/cast", json={"name": "Sam"}, headers=headers).json()["id"]

    response = client.post(f"/cast/{person_id}/train", json={"confirm": False}, headers=headers)

    assert response.status_code == 422


def test_train_422s_without_usable_photos_even_with_confirm(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]], arq_pool: AsyncMock
) -> None:
    headers = auth_headers("cast-user-5")
    person_id = client.post("/cast", json={"name": "Sam"}, headers=headers).json()["id"]

    response = client.post(f"/cast/{person_id}/train", json={"confirm": True}, headers=headers)

    assert response.status_code == 422
    arq_pool.enqueue_job.assert_not_called()


def test_train_only_enqueues_once_and_reports_progress(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    """A second `/train` while one is pending or running must not enqueue again (guards against a
    real double-spend), and `GET /cast` must expose the training state so the UI can show real
    progress."""
    headers = auth_headers("cast-user-train")
    person_id = client.post("/cast", json={"name": "Sam"}, headers=headers).json()["id"]
    conn.execute(
        "INSERT INTO identity_models (person_id, source_photo_count) VALUES (%s, 5)", (person_id,)
    )
    conn.commit()

    first = client.post(f"/cast/{person_id}/train", json={"confirm": True}, headers=headers)
    second = client.post(f"/cast/{person_id}/train", json={"confirm": True}, headers=headers)

    assert first.status_code == second.status_code == 202
    assert arq_pool.enqueue_job.call_count == 1

    person = client.get("/cast", headers=headers).json()["people"][0]
    assert person["training"]["status"] == "pending"
    assert person["training"]["stage"] is None

    # Once it has finished, training again is allowed (a real retry).
    conn.execute("UPDATE jobs SET status = 'failed' WHERE id = %s", (f"train:{person_id}",))
    conn.commit()
    client.post(f"/cast/{person_id}/train", json={"confirm": True}, headers=headers)
    assert arq_pool.enqueue_job.call_count == 2


def test_revoke_deletes_the_person_row(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    headers = auth_headers("cast-user-6")
    person_id = client.post("/cast", json={"name": "Sam"}, headers=headers).json()["id"]

    response = client.delete(f"/cast/{person_id}", headers=headers)

    assert response.status_code == 200
    row = conn.execute("SELECT id FROM people WHERE id = %s", (person_id,)).fetchone()
    assert row is None


def test_cast_endpoints_404_for_someone_elses_person(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    owner_headers = auth_headers("cast-owner")
    person_id = client.post("/cast", json={"name": "Sam"}, headers=owner_headers).json()["id"]

    response = client.post(
        f"/cast/{person_id}/photos/presign",
        json={"filename": "x.jpg"},
        headers=auth_headers("cast-stranger"),
    )

    assert response.status_code == 404


def test_invite_link_round_trips_through_verify_invite(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    from app.routers.cast import verify_invite

    response = client.post("/cast/invite", json={}, headers=auth_headers("inviter"))

    assert response.status_code == 200
    assert verify_invite(response.json()["token"]) == "inviter"


def test_verify_invite_rejects_a_tampered_token() -> None:
    from fastapi import HTTPException

    from app.routers.cast import verify_invite

    with pytest.raises(HTTPException):
        verify_invite("someone:9999999999:not-a-real-signature")


def test_pick_master_promotes_the_chosen_candidate_and_clears_the_rest(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The person picks their own master from the candidates `train_character_job` saved, instead
    of it auto-approving #1."""
    import json
    from pathlib import Path

    from storage import LocalFileStorage

    import app.deps as deps_module

    storage = LocalFileStorage(root=Path(str(tmp_path)), base_url="http://testserver")
    monkeypatch.setattr(deps_module, "get_storage", lambda: storage)

    headers = auth_headers("cast-user-master")
    person_id = client.post("/cast", json={"name": "Sam"}, headers=headers).json()["id"]
    urls = {
        cid: storage.put_object(
            f"people/{person_id}/master_candidates/{cid}.png",
            cid.encode(),
            content_type="image/png",
        )
        for cid in ("a-0", "b-1")
    }
    candidates = [
        {
            "id": cid,
            "url": url,
            "seed": 7,
            "rank_score": 0.9,
            "style_score": 0.5,
            "checklist_score": 1.0,
            "style_card_version": "v2",
        }
        for cid, url in urls.items()
    ]
    conn.execute(
        "INSERT INTO identity_models (person_id, source_photo_count, master_candidates) "
        "VALUES (%s, 5, %s::jsonb)",
        (person_id, json.dumps(candidates)),
    )
    conn.commit()

    listing = client.get("/cast", headers=headers).json()["people"][0]
    assert [c["id"] for c in listing["identity"]["master_candidates"]] == ["a-0", "b-1"]

    missing = client.post(
        f"/cast/{person_id}/master", json={"candidate_id": "nope"}, headers=headers
    )
    assert missing.status_code == 404

    picked = client.post(f"/cast/{person_id}/master", json={"candidate_id": "b-1"}, headers=headers)
    assert picked.status_code == 200
    assert storage.read_path(f"people/{person_id}/master.png").read_bytes() == b"b-1"
    assert not storage.read_path(f"people/{person_id}/master_candidates/a-0.png").exists()

    after = client.get("/cast", headers=headers).json()["people"][0]["identity"]
    assert after["master_path"].endswith(f"people/{person_id}/master.png")
    assert after["master_candidates"] is None
    assert after["generation_path"] == "reference"


def test_age_is_saved_up_front_patched_independently_and_validated(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    """Age is asked alongside gender (setup step 1) and stored on the person; the two are patched
    independently."""
    headers = auth_headers("cast-user-age")
    created = client.post("/cast", json={"name": "Sam", "age": 34}, headers=headers)
    assert created.status_code == 201
    assert created.json()["age"] == 34
    person_id = created.json()["id"]

    # Setting only gender must not wipe the age (and vice versa).
    client.patch(f"/cast/{person_id}", json={"gender_term": "man"}, headers=headers)
    person = client.get("/cast", headers=headers).json()["people"][0]
    assert (person["gender_term"], person["age"]) == ("man", 34)

    client.patch(f"/cast/{person_id}", json={"age": 35}, headers=headers)
    person = client.get("/cast", headers=headers).json()["people"][0]
    assert (person["gender_term"], person["age"]) == ("man", 35)

    for bad in (0, 121, -5):
        response = client.patch(f"/cast/{person_id}", json={"age": bad}, headers=headers)
        assert response.status_code == 422
