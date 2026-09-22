"""PostgresMemoryStore — the real pgvector-backed memory store. Runs
against a real local Postgres (`DATABASE_URL`) in a dedicated
`comiccanvas_test` schema so it never touches dev data, using the real
`bge-small-en-v1.5` model, already cached locally so this doesn't need
network for the model; `enable_socket` is for the real Postgres TCP
connection.

Skips if `DATABASE_URL` isn't configured, mirroring
`test_storage_contract.py`'s R2-credentials skip, so this suite is
still runnable somewhere without a local Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.db import drop_schema
from app.memory import PostgresMemoryStore
from contracts import Beat, BeatSheet

pytestmark = pytest.mark.enable_socket

TEST_SCHEMA = "comiccanvas_test"


def _require_database() -> None:
    try:
        get_settings()
    except ValidationError:
        pytest.skip("DATABASE_URL not configured; see the README for local Postgres setup")


@pytest.fixture
def store() -> Iterator[PostgresMemoryStore]:
    _require_database()
    drop_schema(TEST_SCHEMA)
    yield PostgresMemoryStore(schema=TEST_SCHEMA)
    drop_schema(TEST_SCHEMA)


def _beat(beat_id: str, event: str, *, place: str = "gym", quote: str | None = None) -> Beat:
    return Beat(
        id=beat_id,
        time="evening",
        place=place,
        event=event,
        emotion="content",
        importance=0.5,
        humor=0.2,
        quote=quote,
    )


def _sheet(d: date, beats: list[Beat], people: list[str] | None = None) -> BeatSheet:
    return BeatSheet(date=d, mood_arc=["steady"], beats=beats, people_mentioned=people or [])


def test_write_beats_persists_known_people_and_places(store: PostgresMemoryStore) -> None:
    store.write_beats(
        "u1",
        _sheet(date(2026, 9, 3), [_beat("b1", "first gym session in two weeks")], people=["Sam"]),
    )

    assert store.known_people("u1") == ["Sam"]
    assert store.known_places("u1") == ["gym"]


def test_write_beats_is_idempotent_on_conflict(store: PostgresMemoryStore) -> None:
    sheet = _sheet(date(2026, 9, 3), [_beat("b1", "first gym session")])
    store.write_beats("u1", sheet)
    store.write_beats("u1", sheet)  # re-run of the same job

    results = store.search("u1", "gym", k=10)
    assert len([r for r in results if r.beat.id == "b1"]) == 1


def test_recent_people_respects_the_days_window(store: PostgresMemoryStore) -> None:
    today = date.today()
    store.write_beats(
        "u1", _sheet(today - timedelta(days=1), [_beat("b1", "coffee")], people=["Sam"])
    )
    store.write_beats(
        "u1", _sheet(today - timedelta(days=10), [_beat("b2", "coffee")], people=["Alex"])
    )

    assert store.recent_people("u1", days=3) == ["Sam"]


def test_place_ref_round_trips(store: PostgresMemoryStore) -> None:
    assert store.place_ref("u1", "gym") is None

    store.set_place_ref("u1", "gym", "https://example.test/gym.png")

    assert store.place_ref("u1", "gym") == "https://example.test/gym.png"


def test_known_vocab_aggregates_people_places_and_objects(store: PostgresMemoryStore) -> None:
    beat = _beat("b1", "packed the bag before the trip")
    beat = beat.model_copy(update={"objects": ["backpack", "passport"]})
    store.write_beats("u1", _sheet(date(2026, 9, 3), [beat], people=["Sam"]))

    vocab = store.known_vocab("u1", n=40)

    assert set(vocab) >= {"Sam", "gym", "backpack", "passport"}


def test_search_gym_query_returns_the_sep_3_beat_first(store: PostgresMemoryStore) -> None:
    """A "gym" query on the seed fixture returns the Sep 3 beat first.
    Real embeddings, real pgvector cosine ranking — not a keyword-overlap
    approximation."""
    store.write_beats(
        "u1",
        _sheet(
            date(2026, 9, 1),
            [_beat("b_sep1", "walked to the cafe and read a book by the window", place="cafe")],
        ),
    )
    store.write_beats(
        "u1",
        _sheet(
            date(2026, 9, 2),
            [_beat("b_sep2", "cooked pasta for dinner and watched a movie", place="kitchen")],
        ),
    )
    store.write_beats(
        "u1",
        _sheet(
            date(2026, 9, 3),
            [_beat("b_sep3", "first gym session in two weeks, legs sore afterward", place="gym")],
        ),
    )
    store.write_beats(
        "u1",
        _sheet(
            date(2026, 9, 4),
            [_beat("b_sep4", "long commute on the train, read the news", place="transit")],
        ),
    )

    results = store.search("u1", "gym", k=3)

    assert results[0].beat.id == "b_sep3"


def test_search_attaches_panel_urls_written_alongside_the_beat(store: PostgresMemoryStore) -> None:
    store.write_beats(
        "u1",
        _sheet(date(2026, 9, 3), [_beat("b1", "first gym session in two weeks")]),
        panel_urls={"b1": ["https://example.test/p1.png"]},
    )

    results = store.search("u1", "gym", k=1)

    assert results[0].panel_urls == ["https://example.test/p1.png"]


def test_search_is_scoped_per_user(store: PostgresMemoryStore) -> None:
    store.write_beats("u1", _sheet(date(2026, 9, 3), [_beat("b1", "first gym session")]))

    assert store.search("u2", "gym", k=5) == []


def test_top_rated_scripts_is_honestly_empty_before_phase_9(store: PostgresMemoryStore) -> None:
    assert store.top_rated_scripts("u1") == []


def test_same_beat_id_on_different_days_does_not_overwrite(store: PostgresMemoryStore) -> None:
    """`Beat.id` only has to be unique WITHIN one day's `BeatSheet` — the
    extractor restarts numbering ("b1", "b2", ...) every day, so the
    `memory_beats` constraint must be `UNIQUE (user_id, date, beat_id)`,
    not `UNIQUE (user_id, beat_id)` alone, or a second day's "b1" would
    silently overwrite the first day's."""
    store.write_beats("u1", _sheet(date(2026, 3, 1), [_beat("b1", "monday event")]))
    store.write_beats("u1", _sheet(date(2026, 3, 2), [_beat("b1", "tuesday event")]))

    monday_results = store.search("u1", "monday", k=10)
    tuesday_results = store.search("u1", "tuesday", k=10)

    assert any(r.beat.event == "monday event" for r in monday_results)
    assert any(r.beat.event == "tuesday event" for r in tuesday_results)


def test_get_panel_urls_is_scoped_by_day_not_just_beat_id(store: PostgresMemoryStore) -> None:
    store.write_beats(
        "u1",
        _sheet(date(2026, 3, 1), [_beat("b1", "monday event")]),
        panel_urls={"b1": ["https://example.test/monday.png"]},
    )
    store.write_beats(
        "u1",
        _sheet(date(2026, 3, 2), [_beat("b1", "tuesday event")]),
        panel_urls={"b1": ["https://example.test/tuesday.png"]},
    )

    assert store.get_panel_urls("u1", "b1", date(2026, 3, 1)) == ["https://example.test/monday.png"]
    assert store.get_panel_urls("u1", "b1", date(2026, 3, 2)) == [
        "https://example.test/tuesday.png"
    ]
    assert store.get_panel_urls("u1", "b1", date(2026, 3, 3)) == []
