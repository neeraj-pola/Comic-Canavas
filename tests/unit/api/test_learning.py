"""`GET /learning` — the preference-learning section: the model state,
one snapshot per tap, and the exploration rate, all read from what the
worker wrote after each A/B tap."""

from __future__ import annotations

import json
from collections.abc import Callable

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow

pytestmark = pytest.mark.enable_socket

FEATURES = [
    "identity",
    "style",
    "alignment",
    "detail",
    "warmth",
    "brightness",
    "saturation",
    "contrast",
    "closeness",
    "expression",
]


def _zeros() -> str:
    return json.dumps([0.0] * len(FEATURES))


def test_learning_has_an_empty_preference_section_before_any_tap(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    body = client.get("/learning", headers=auth_headers("learn-none")).json()
    pref = body["preference"]
    assert pref["n_taps"] == 0 and pref["active"] is False
    assert pref["snapshots"] == [] and pref["explore_by_day"] == []


def test_learning_returns_the_model_snapshots_and_exploration_rate(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    user = "learn-user"
    conn.execute("INSERT INTO users (id) VALUES (%s)", (user,))
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES ('j1', %s, 'daily', 'done')", (user,)
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source) VALUES ('j1', %s, '2026-09-19', 'text')",
        (user,),
    )
    conn.execute(
        "INSERT INTO beats (day_id, beat_id, time, place, event, emotion, importance, humor) "
        "VALUES ('j1','b1','morning','desk','e','calm',0.5,0.5)"
    )
    conn.execute(
        "INSERT INTO panels (day_id, panel_id, beat_id, place, time_of_day, expression, action, "
        "framing, caption_a) VALUES ('j1',1,'b1','desk','morning','neutral','a','wide','c')"
    )
    for cid, explored in (("c1", 1.0), ("c2", 0.0)):
        conn.execute(
            "INSERT INTO candidates (id, day_id, panel_id, url, seed, scores, chosen) "
            "VALUES (%s,'j1',1,%s,1,%s::jsonb,%s)",
            (cid, f"http://x/{cid}", json.dumps({"explored": explored}), cid == "c1"),
        )
    conn.execute(
        "INSERT INTO preference_models "
        "(user_id, features, mu, cov, scales, n_taps, active, lean, z, accuracy) "
        "VALUES (%s, %s::jsonb, %s::jsonb, '[]'::jsonb, '[]'::jsonb, 1, true, "
        "%s::jsonb, %s::jsonb, %s::jsonb)",
        (
            user,
            json.dumps(FEATURES),
            _zeros(),
            json.dumps({"warmth": 1}),
            json.dumps({"warmth": 2.1}),
            json.dumps({"learned": 0.8, "hand": 0.5}),
        ),
    )
    conn.execute(
        "INSERT INTO preference_snapshots (user_id, tap_index, day_id, panel_id, chosen_url,"
        " rejected_url, prob, correct, hand_correct, mu, sd, delta, axis)"
        " VALUES (%s,1,'j1',1,'http://x/c1','http://x/c2',0.7,1,0.5,%s::jsonb,%s::jsonb,%s::jsonb,0)",
        (user, _zeros(), _zeros(), _zeros()),
    )
    conn.commit()

    pref = client.get("/learning", headers=auth_headers(user)).json()["preference"]

    assert pref["n_taps"] == 1 and pref["active"] is True
    assert pref["lean"] == {"warmth": 1} and pref["accuracy"]["learned"] == 0.8
    (snap,) = pref["snapshots"]
    assert snap["tap"] == 1 and snap["date"] == "2026-09-19" and snap["axis"] == 0
    assert snap["prob"] == 0.7 and len(snap["mu"]) == len(FEATURES)
    assert pref["explore_by_day"] == [{"date": "2026-09-19", "panels": 1, "explored": 1}]

    # Another user never sees these.
    other = client.get("/learning", headers=auth_headers("someone-else")).json()["preference"]
    assert other["snapshots"] == []


def _seed_day_with_panels(
    conn: psycopg.Connection[DictRow],
    user: str,
    job: str,
    date: str,
    rewards: dict[int, tuple[float, float]],
) -> None:
    """One day; per panel two candidates (A, B) whose stored `reward` is `rewards[panel]`."""
    conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (user,))
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s,%s,'daily','done')", (job, user)
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source) VALUES (%s,%s,%s,'text')",
        (job, user, date),
    )
    conn.execute(
        "INSERT INTO beats (day_id, beat_id, time, place, event, emotion, importance, humor) "
        "VALUES (%s,'b1','morning','desk','e','calm',0.5,0.5)",
        (job,),
    )
    for panel, (reward_a, reward_b) in rewards.items():
        conn.execute(
            "INSERT INTO panels (day_id, panel_id, beat_id, place, time_of_day, expression, "
            "action, framing, caption_a) "
            "VALUES (%s,%s,'b1','desk','morning','neutral','a','wide','c')",
            (job, panel),
        )
        for name, reward in (("a", reward_a), ("b", reward_b)):
            conn.execute(
                "INSERT INTO candidates (id, day_id, panel_id, url, seed, scores) "
                "VALUES (%s,%s,%s,%s,1,%s::jsonb)",
                (
                    f"{job}-{panel}{name}",
                    job,
                    panel,
                    f"http://x/{job}-{panel}{name}",
                    json.dumps({"reward": reward}),
                ),
            )


def _tap(
    conn: psycopg.Connection[DictRow],
    job: str,
    panel: int,
    chosen: str,
    rejected: str,
    *,
    minute: int,
    source: str = "ab",
) -> None:
    """`minute` gives each tap its own time — real taps arrive in separate requests, so
    "the latest tap for a panel" is well defined (one transaction would share `now()`)."""
    conn.execute(
        "INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source, created_at) "
        "VALUES (%s, %s, %s, %s, %s, "
        "timestamptz '2026-09-14 10:00+00' + make_interval(mins => %s))",
        (
            job,
            panel,
            f"http://x/{job}-{panel}{chosen}",
            f"http://x/{job}-{panel}{rejected}",
            source,
            minute,
        ),
    )


def test_first_choice_agreement_and_tap_count_are_computed_not_guessed(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    """The "first-pick rate" can't order a panel's candidates by `created_at` (they all share one
    timestamp), so it's computed from stored reward ranking instead; the "pairs" tile counts
    distinct taps, not every `image_pairs` row (a changed mind, regenerations)."""
    user, job = "learn-agree", "learn-agree:2026-09-14"
    # panel 1: A ranked higher; panel 2: B ranked higher; panel 3: exactly tied
    _seed_day_with_panels(
        conn, user, job, "2026-09-14", {1: (0.6, 0.4), 2: (0.3, 0.5), 3: (0.5, 0.5)}
    )
    _tap(conn, job, 1, "b", "a", minute=1)  # picked the LOWER-ranked one: disagrees with the app
    _tap(
        conn, job, 1, "a", "b", minute=2
    )  # ...then changed their mind: only this LATEST tap counts
    _tap(conn, job, 2, "a", "b", minute=3)  # picked the lower-ranked one: disagrees
    _tap(conn, job, 3, "a", "b", minute=4)  # a tie counts half
    _tap(conn, job, 2, "b", "a", minute=5, source="regenerate")  # not a tap at all
    conn.commit()

    body = client.get("/learning", headers=auth_headers(user)).json()

    assert body["taps"] == 3 and body["pairs"] == 3  # distinct panels, not 5 rows
    assert body["regenerations"] == 1
    (day,) = body["pick_rate_series"]
    assert day["day_id"] == "2026-09-14" and day["taps"] == 3
    assert day["pick_rate"] == pytest.approx((1 + 0 + 0.5) / 3)  # agree (latest tap), disagree, tie


def test_first_choice_agreement_reads_a_best_of_three_pick_as_one_tap(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    """A best-of-3 tap wrote one row per option passed on under a shared `tap_id`; it is ONE tap,
    and it agrees with the app when the pick was the highest-ranked of all three shown."""
    user, job = "learn-b3", "learn-b3:2026-09-15"
    _seed_day_with_panels(conn, user, job, "2026-09-15", {1: (0.6, 0.4), 2: (0.3, 0.5)})
    for panel, reward in ((1, 0.5), (2, 0.2)):  # a third option per panel
        conn.execute(
            "INSERT INTO candidates (id, day_id, panel_id, url, seed, scores) "
            "VALUES (%s,%s,%s,%s,1,%s::jsonb)",
            (
                f"{job}-{panel}c",
                job,
                panel,
                f"http://x/{job}-{panel}c",
                json.dumps({"reward": reward}),
            ),
        )
    for panel, picked, others in ((1, "a", ("b", "c")), (2, "a", ("b", "c"))):
        for other in others:
            conn.execute(
                "INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source, tap_id) "
                "VALUES (%s,%s,%s,%s,'ab',%s)",
                (
                    job,
                    panel,
                    f"http://x/{job}-{panel}{picked}",
                    f"http://x/{job}-{panel}{other}",
                    f"tap-{panel}",
                ),
            )
    conn.commit()

    body = client.get("/learning", headers=auth_headers(user)).json()

    assert body["taps"] == 2  # two taps, not four rows
    (day,) = body["pick_rate_series"]
    # panel 1: picked a (0.6), the best of 0.6/0.4/0.5 -> agrees;
    # panel 2: picked a (0.3) but b (0.5) ranked higher -> disagrees
    assert day["taps"] == 2 and day["pick_rate"] == pytest.approx(0.5)


def test_default_pick_rate_is_reported_per_day_with_unknowns_left_out(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    user, job = "learn-kpi", "learn-kpi:2026-09-19"
    _seed_day_with_panels(conn, user, job, "2026-09-19", {1: (0.6, 0.4)})
    conn.execute(
        "INSERT INTO preference_models "
        "(user_id, features, mu, cov, scales, n_taps, active) "
        "VALUES (%s, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, 3, false)",
        (user,),
    )
    for tap, default_picked in ((1, True), (2, False), (3, None)):
        conn.execute(
            "INSERT INTO preference_snapshots (user_id, tap_index, day_id, panel_id, chosen_url,"
            " rejected_url, prob, correct, hand_correct, mu, sd, delta, default_picked)"
            " VALUES (%s,%s,%s,1,'a','b',0.5,1,0.5,'[]'::jsonb,'[]'::jsonb,'[]'::jsonb,%s)",
            (user, tap, job, default_picked),
        )
    conn.commit()

    pref = client.get("/learning", headers=auth_headers(user)).json()["preference"]

    assert pref["default_pick_by_day"] == [{"day_id": "2026-09-19", "pick_rate": 0.5, "taps": 2}]
    assert pref["knobs"] == {} and pref["notes"] == {}


def test_ratings_by_day_counts_the_latest_rating_per_image(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    user, job = "learn-rate", "learn-rate:2026-09-21"
    _seed_day_with_panels(conn, user, job, "2026-09-21", {1: (0.6, 0.4), 2: (0.3, 0.5)})
    plan = [(1, "a", 0), (1, "a", 2), (1, "b", 1), (2, "a", 1), (2, "b", 0)]  # panel, image, rating
    for panel, image, rating in plan:
        conn.execute(
            "INSERT INTO image_ratings (day_id, panel_id, url, rating) VALUES (%s,%s,%s,%s)",
            (job, panel, f"http://x/{job}-{panel}{image}", rating),
        )
    conn.commit()

    body = client.get("/learning", headers=auth_headers(user)).json()

    # panel 1 image a: off, then changed to great (only the latest counts)
    assert body["ratings_by_day"] == [{"day_id": "2026-09-21", "off": 1, "ok": 2, "great": 1}]
    assert client.get("/learning", headers=auth_headers("nobody")).json()["ratings_by_day"] == []
