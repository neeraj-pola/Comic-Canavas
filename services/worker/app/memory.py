"""Memory node — pgvector-backed store for beats, people and places, plus an
in-memory backend for tests.

One `MemoryStore` Protocol, two implementations, the same shape as
`packages/storage`'s Local/R2 split. `InMemoryMemoryStore` is the default for
unit tests that don't need a database; `PostgresMemoryStore` is the real
backend, using `bge-small-en-v1.5` embeddings (`app/embeddings.py`) over
pgvector cosine distance for beats/people/place-ref persistence.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol

from pydantic import BaseModel

from contracts import Beat, BeatSheet, Script


class ScriptExample(BaseModel):
    """A past (beat sheet, script) pair good enough to few-shot from, ranked
    by thumbs/pick data. `InMemoryMemoryStore` tracks a `rating` directly."""

    beat_sheet: BeatSheet
    script: Script


class MemorySearchResult(BaseModel):
    """A search hit: a beat, its relevance score, and the panel URLs recorded
    for it via `write_beats(..., panel_urls=...)`."""

    beat: Beat
    score: float
    panel_urls: list[str] = []


class MemoryStore(Protocol):
    def write_beats(
        self,
        user_id: str,
        beat_sheet: BeatSheet,
        *,
        panel_urls: dict[str, list[str]] | None = None,
    ) -> None: ...
    def known_people(self, user_id: str) -> list[str]: ...
    def known_places(self, user_id: str) -> list[str]: ...
    def known_vocab(self, user_id: str, n: int = 40) -> list[str]: ...
    def top_rated_scripts(self, user_id: str, n: int = 10) -> list[ScriptExample]: ...
    def place_ref(self, user_id: str, place: str) -> str | None: ...
    def set_place_ref(self, user_id: str, place: str, ref: str) -> None: ...
    def recent_people(self, user_id: str, days: int) -> list[str]: ...
    def search(self, user_id: str, q: str, k: int = 5) -> list[MemorySearchResult]: ...
    def get_panel_urls(self, user_id: str, beat_id: str, day: date) -> list[str]: ...


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._beats: dict[str, list[BeatSheet]] = {}
        self._people: dict[str, list[str]] = {}
        self._people_last_seen: dict[str, dict[str, date]] = {}
        self._places: dict[str, list[str]] = {}
        self._vocab: dict[str, list[str]] = {}
        self._rated_scripts: dict[str, list[tuple[float, ScriptExample]]] = {}
        self._place_refs: dict[tuple[str, str], str] = {}
        # Keyed by (user_id, date, beat_id) since `Beat.id` is only unique
        # within one day's BeatSheet, not across days.
        self._panel_urls: dict[tuple[str, date, str], list[str]] = {}

    def write_beats(
        self,
        user_id: str,
        beat_sheet: BeatSheet,
        *,
        panel_urls: dict[str, list[str]] | None = None,
    ) -> None:
        self._beats.setdefault(user_id, []).append(beat_sheet)
        for person in beat_sheet.people_mentioned:
            people = self._people.setdefault(user_id, [])
            if person not in people:
                people.append(person)
            last_seen = self._people_last_seen.setdefault(user_id, {})
            if person not in last_seen or beat_sheet.date > last_seen[person]:
                last_seen[person] = beat_sheet.date
        for beat in beat_sheet.beats:
            places = self._places.setdefault(user_id, [])
            if beat.place not in places:
                places.append(beat.place)
            if panel_urls and beat.id in panel_urls:
                self._panel_urls[(user_id, beat_sheet.date, beat.id)] = panel_urls[beat.id]

    def known_people(self, user_id: str) -> list[str]:
        return list(self._people.get(user_id, []))

    def known_places(self, user_id: str) -> list[str]:
        return list(self._places.get(user_id, []))

    def known_vocab(self, user_id: str, n: int = 40) -> list[str]:
        return list(self._vocab.get(user_id, []))[:n]

    def top_rated_scripts(self, user_id: str, n: int = 10) -> list[ScriptExample]:
        rated = sorted(self._rated_scripts.get(user_id, []), key=lambda pair: pair[0], reverse=True)
        return [example for _rating, example in rated[:n]]

    def add_rated_script(self, user_id: str, example: ScriptExample, *, rating: float) -> None:
        """Test/fixture helper for ranking; the real store ranks from thumbs/pick data instead."""
        self._rated_scripts.setdefault(user_id, []).append((rating, example))

    def place_ref(self, user_id: str, place: str) -> str | None:
        """None until this place has a chosen reference panel."""
        return self._place_refs.get((user_id, place))

    def set_place_ref(self, user_id: str, place: str, ref: str) -> None:
        self._place_refs[(user_id, place)] = ref

    def recent_people(self, user_id: str, days: int) -> list[str]:
        cutoff = date.today() - timedelta(days=days)
        last_seen = self._people_last_seen.get(user_id, {})
        return [person for person, seen in last_seen.items() if seen >= cutoff]

    def search(self, user_id: str, q: str, k: int = 5) -> list[MemorySearchResult]:
        """Keyword-overlap fallback, not semantic search — semantic ranking
        only exists in `PostgresMemoryStore`. Lets code that just needs a
        `MemoryStore` be tested without a database."""
        query_words = set(q.lower().split())
        scored: list[tuple[float, Beat, date]] = []
        for sheet in self._beats.get(user_id, []):
            for beat in sheet.beats:
                text = f"{beat.event} {beat.place_detail or ''} {beat.quote or ''}".lower()
                beat_words = set(text.split())
                overlap = len(query_words & beat_words)
                if overlap:
                    scored.append((overlap / max(len(query_words), 1), beat, sheet.date))
        scored.sort(key=lambda triple: triple[0], reverse=True)
        return [
            MemorySearchResult(
                beat=beat,
                score=score,
                panel_urls=self._panel_urls.get((user_id, beat_date, beat.id), []),
            )
            for score, beat, beat_date in scored[:k]
        ]

    def get_panel_urls(self, user_id: str, beat_id: str, day: date) -> list[str]:
        return list(self._panel_urls.get((user_id, day, beat_id), []))

    def seed(
        self,
        user_id: str,
        *,
        people: list[str] | None = None,
        places: list[str] | None = None,
        vocab: list[str] | None = None,
    ) -> None:
        """Test/fixture helper; the real store reads from Postgres instead."""
        if people is not None:
            self._people[user_id] = list(people)
        if places is not None:
            self._places[user_id] = list(places)
        if vocab is not None:
            self._vocab[user_id] = list(vocab)


class PostgresMemoryStore:
    """The real backend. Each method opens its own short-lived connection
    (via `app.db.connect`) rather than holding one open, since this runs
    inside arq worker jobs rather than a long-lived pooled server process."""

    def __init__(self, *, schema: str = "public") -> None:
        self.schema = schema

    def write_beats(
        self,
        user_id: str,
        beat_sheet: BeatSheet,
        *,
        panel_urls: dict[str, list[str]] | None = None,
    ) -> None:
        from pgvector import Vector

        from app.db import connect
        from app.embeddings import embed_passage

        panel_urls = panel_urls or {}
        with connect(schema=self.schema) as conn:
            for beat in beat_sheet.beats:
                text = _embeddable_text(beat)
                embedding = Vector(embed_passage(text))
                conn.execute(
                    """
                    INSERT INTO memory_beats (
                        user_id, beat_id, date, time, place, place_detail, event,
                        emotion, people, objects, importance, humor, quote,
                        panel_urls, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, date, beat_id) DO UPDATE SET
                        time = EXCLUDED.time, place = EXCLUDED.place,
                        place_detail = EXCLUDED.place_detail, event = EXCLUDED.event,
                        emotion = EXCLUDED.emotion, people = EXCLUDED.people,
                        objects = EXCLUDED.objects, importance = EXCLUDED.importance,
                        humor = EXCLUDED.humor, quote = EXCLUDED.quote,
                        panel_urls = EXCLUDED.panel_urls, embedding = EXCLUDED.embedding
                    """,
                    (
                        user_id,
                        beat.id,
                        beat_sheet.date,
                        beat.time,
                        beat.place,
                        beat.place_detail,
                        beat.event,
                        beat.emotion,
                        beat.people,
                        beat.objects,
                        beat.importance,
                        beat.humor,
                        beat.quote,
                        panel_urls.get(beat.id, []),
                        embedding,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO memory_places (user_id, place) VALUES (%s, %s)
                    ON CONFLICT (user_id, place) DO NOTHING
                    """,
                    (user_id, beat.place),
                )
            for person in beat_sheet.people_mentioned:
                conn.execute(
                    """
                    INSERT INTO memory_people (user_id, person, last_seen) VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, person) DO UPDATE
                        SET last_seen = GREATEST(memory_people.last_seen, EXCLUDED.last_seen)
                    """,
                    (user_id, person, beat_sheet.date),
                )

    def known_people(self, user_id: str) -> list[str]:
        from app.db import connect

        with connect(schema=self.schema) as conn:
            rows = conn.execute(
                "SELECT person FROM memory_people WHERE user_id = %s ORDER BY last_seen DESC",
                (user_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def known_places(self, user_id: str) -> list[str]:
        from app.db import connect

        with connect(schema=self.schema) as conn:
            rows = conn.execute(
                "SELECT place FROM memory_places WHERE user_id = %s ORDER BY place", (user_id,)
            ).fetchall()
        return [row[0] for row in rows]

    def known_vocab(self, user_id: str, n: int = 40) -> list[str]:
        """Entities a transcription prompt benefits from knowing about: known
        people, known places, and distinct objects mentioned across this
        user's beats."""
        from app.db import connect

        with connect(schema=self.schema) as conn:
            people = [
                r[0]
                for r in conn.execute(
                    "SELECT person FROM memory_people WHERE user_id = %s", (user_id,)
                ).fetchall()
            ]
            places = [
                r[0]
                for r in conn.execute(
                    "SELECT place FROM memory_places WHERE user_id = %s", (user_id,)
                ).fetchall()
            ]
            objects_rows = conn.execute(
                "SELECT DISTINCT unnest(objects) FROM memory_beats WHERE user_id = %s", (user_id,)
            ).fetchall()
        objects = [r[0] for r in objects_rows]
        vocab = sorted(set(people) | set(places) | set(objects))
        return vocab[:n]

    def top_rated_scripts(self, user_id: str, n: int = 10) -> list[ScriptExample]:
        """Returns nothing until real thumbs/pick-data ranking is implemented."""
        return []

    def place_ref(self, user_id: str, place: str) -> str | None:
        from app.db import connect

        with connect(schema=self.schema) as conn:
            row = conn.execute(
                "SELECT ref_url FROM memory_places WHERE user_id = %s AND place = %s",
                (user_id, place),
            ).fetchone()
        return row[0] if row else None

    def set_place_ref(self, user_id: str, place: str, ref: str) -> None:
        from app.db import connect

        with connect(schema=self.schema) as conn:
            conn.execute(
                """
                INSERT INTO memory_places (user_id, place, ref_url) VALUES (%s, %s, %s)
                ON CONFLICT (user_id, place) DO UPDATE SET ref_url = EXCLUDED.ref_url
                """,
                (user_id, place, ref),
            )

    def recent_people(self, user_id: str, days: int) -> list[str]:
        from app.db import connect

        with connect(schema=self.schema) as conn:
            rows = conn.execute(
                """
                SELECT person FROM memory_people
                WHERE user_id = %s AND last_seen >= (CURRENT_DATE - %s::int)
                ORDER BY last_seen DESC
                """,
                (user_id, days),
            ).fetchall()
        return [row[0] for row in rows]

    def search(self, user_id: str, q: str, k: int = 5) -> list[MemorySearchResult]:
        from pgvector import Vector

        from app.db import connect
        from app.embeddings import embed_query

        query_embedding = Vector(embed_query(q))
        with connect(schema=self.schema) as conn:
            rows = conn.execute(
                """
                SELECT beat_id, date, time, place, place_detail, event, emotion,
                       people, objects, importance, humor, quote, panel_urls,
                       1 - (embedding <=> %s) AS score
                FROM memory_beats
                WHERE user_id = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, user_id, query_embedding, k),
            ).fetchall()

        results = []
        for row in rows:
            (
                beat_id,
                _beat_date,
                time,
                place,
                place_detail,
                event,
                emotion,
                people,
                objects,
                importance,
                humor,
                quote,
                panel_urls,
                score,
            ) = row
            beat = Beat(
                id=beat_id,
                time=time,
                place=place,
                place_detail=place_detail,
                event=event,
                emotion=emotion,
                people=list(people or []),
                objects=list(objects or []),
                importance=importance,
                humor=humor,
                quote=quote,
            )
            results.append(
                MemorySearchResult(beat=beat, score=score, panel_urls=list(panel_urls or []))
            )
        return results

    def get_panel_urls(self, user_id: str, beat_id: str, day: date) -> list[str]:
        """Does this specific day's beat already have a chosen panel? Keyed by
        `(user_id, date, beat_id)`, since `beat_id` alone is only unique
        within one day's `BeatSheet`."""
        from app.db import connect

        with connect(schema=self.schema) as conn:
            row = conn.execute(
                "SELECT panel_urls FROM memory_beats "
                "WHERE user_id = %s AND date = %s AND beat_id = %s",
                (user_id, day, beat_id),
            ).fetchone()
        return list(row[0]) if row and row[0] else []


def _embeddable_text(beat: Beat) -> str:
    parts = [beat.event]
    if beat.place_detail:
        parts.append(beat.place_detail)
    if beat.quote:
        parts.append(beat.quote)
    return ". ".join(parts)


_store: MemoryStore = InMemoryMemoryStore()


def get_memory_store() -> MemoryStore:
    return _store


def set_memory_store(store: MemoryStore) -> None:
    global _store
    _store = store
