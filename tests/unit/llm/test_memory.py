"""app/memory.py's `InMemoryMemoryStore` — the interim, database-free
backend most tests use. `PostgresMemoryStore` (the real
backend) is exercised separately in `test_memory_postgres.py` against a
real local Postgres, since a keyword-overlap fallback can't verify
anything about real semantic search.
"""

from __future__ import annotations

from datetime import date, timedelta

from app.memory import InMemoryMemoryStore
from contracts import Beat, BeatSheet


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


def test_write_beats_tracks_known_people_and_places() -> None:
    store = InMemoryMemoryStore()
    store.write_beats(
        "u1", _sheet(date(2026, 9, 3), [_beat("b1", "went to the gym")], people=["Sam"])
    )

    assert store.known_people("u1") == ["Sam"]
    assert store.known_places("u1") == ["gym"]


def test_recent_people_respects_the_days_window() -> None:
    store = InMemoryMemoryStore()
    today = date.today()
    store.write_beats(
        "u1", _sheet(today - timedelta(days=1), [_beat("b1", "coffee")], people=["Sam"])
    )
    store.write_beats(
        "u1", _sheet(today - timedelta(days=10), [_beat("b2", "coffee")], people=["Alex"])
    )

    recent = store.recent_people("u1", days=3)

    assert recent == ["Sam"]


def test_place_ref_defaults_to_none_and_can_be_set() -> None:
    store = InMemoryMemoryStore()
    assert store.place_ref("u1", "gym") is None

    store.set_place_ref("u1", "gym", "https://example.test/gym.png")

    assert store.place_ref("u1", "gym") == "https://example.test/gym.png"


def test_search_ranks_the_matching_beat_first() -> None:
    store = InMemoryMemoryStore()
    store.write_beats(
        "u1",
        _sheet(
            date(2026, 9, 3),
            [_beat("b1", "first gym session in two weeks", place="gym")],
        ),
    )
    store.write_beats(
        "u1",
        _sheet(
            date(2026, 9, 4),
            [_beat("b2", "walked to the cafe and read a book", place="cafe")],
        ),
    )

    results = store.search("u1", "gym")

    assert results[0].beat.id == "b1"


def test_search_attaches_panel_urls_written_alongside_the_beat() -> None:
    store = InMemoryMemoryStore()
    store.write_beats(
        "u1",
        _sheet(date(2026, 9, 3), [_beat("b1", "gym session")]),
        panel_urls={"b1": ["https://example.test/p1.png"]},
    )

    results = store.search("u1", "gym")

    assert results[0].panel_urls == ["https://example.test/p1.png"]


def test_search_scoped_per_user() -> None:
    store = InMemoryMemoryStore()
    store.write_beats("u1", _sheet(date(2026, 9, 3), [_beat("b1", "gym session")]))

    results = store.search("u2", "gym")

    assert results == []


def test_same_beat_id_on_different_days_does_not_overwrite() -> None:
    """`Beat.id` only has to be unique WITHIN one day's `BeatSheet` — the
    extractor restarts numbering ("b1", "b2", ...) every day — so a naive
    `(user_id, beat_id)` key would let one day's beat silently overwrite
    another's. Both must survive here."""
    store = InMemoryMemoryStore()
    store.write_beats("u1", _sheet(date(2026, 3, 1), [_beat("b1", "monday event")]))
    store.write_beats("u1", _sheet(date(2026, 3, 2), [_beat("b1", "tuesday event")]))

    monday_results = store.search("u1", "monday")
    tuesday_results = store.search("u1", "tuesday")

    assert any(r.beat.event == "monday event" for r in monday_results)
    assert any(r.beat.event == "tuesday event" for r in tuesday_results)


def test_get_panel_urls_is_scoped_by_day_not_just_beat_id() -> None:
    store = InMemoryMemoryStore()
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
