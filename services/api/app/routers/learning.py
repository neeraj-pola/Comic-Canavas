"""`GET /learning` — pick-rate series, pairs, caption edits, checkpoints,
next run. Reads whatever is really in `checkpoints`/`training_runs`
(typically empty until real training has run) rather than fabricating
placeholder numbers.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends

from app.db import fetch_one, get_conn
from app.deps import current_user_id

router = APIRouter(prefix="/learning", tags=["learning"])


def _next_sunday_0200() -> str:
    today = date.today()
    days_ahead = (6 - today.weekday()) % 7 or 7  # Sunday=6; always the *next* one
    return (today + timedelta(days=days_ahead)).isoformat() + "T02:00:00"


def _preference(user_id: str) -> dict[str, Any]:
    """The person's learned taste model and, per tap, what it predicted and how the weights
    moved — everything the Learning page's "how it's learning" charts replay. Written by the
    worker after each A/B tap (`app.preference.retrain`); empty until the first tap."""
    with get_conn() as conn:
        model_row = conn.execute(
            "SELECT features, n_taps, active, lean, z, accuracy, knobs, notes, updated_at "
            "FROM preference_models WHERE user_id = %s",
            (user_id,),
        ).fetchone()
        snapshots = conn.execute(
            """
            SELECT s.tap_index AS tap, d.date, s.panel_id, s.chosen_url, s.rejected_url,
                   s.rejected_urls, s.default_picked, s.prob, s.correct, s.hand_correct,
                   s.mu, s.sd, s.delta, s.axis
            FROM preference_snapshots s LEFT JOIN days d ON d.job_id = s.day_id
            WHERE s.user_id = %s ORDER BY s.tap_index
            """,
            (user_id,),
        ).fetchall()
        explore = conn.execute(
            """
            SELECT d.date, count(*) AS panels,
                   count(*) FILTER (WHERE (c.scores->>'explored')::float >= 0.5) AS explored
            FROM candidates c JOIN days d ON d.job_id = c.day_id
            WHERE d.user_id = %s AND d.kind = 'daily' AND c.chosen AND c.scores ? 'explored'
            GROUP BY d.date ORDER BY d.date
            """,
            (user_id,),
        ).fetchall()
    return {
        "features": model_row["features"] if model_row else [],
        "n_taps": model_row["n_taps"] if model_row else 0,
        "active": bool(model_row["active"]) if model_row else False,
        "lean": model_row["lean"] if model_row else {},
        "z": model_row["z"] if model_row else {},
        "accuracy": model_row["accuracy"] if model_row else {},
        "knobs": model_row["knobs"] if model_row else {},
        "notes": model_row["notes"] if model_row else {},
        "default_pick_by_day": _default_pick_by_day([dict(s) for s in snapshots]),
        "updated_at": model_row["updated_at"] if model_row else None,
        "snapshots": [dict(s) for s in snapshots],
        "explore_by_day": [dict(e) for e in explore],
    }


def _default_pick_by_day(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The KPI: per day, how often the person picked the DEFAULT option (the one with no step
    either side of their default) among the options shown. It should climb above the chance
    line (1 / options shown, a third for three) as the defaults get closer to their taste."""
    by_day: dict[str, list[float]] = {}
    for s in snapshots:
        if s["default_picked"] is None or s["date"] is None:
            continue
        by_day.setdefault(s["date"].isoformat(), []).append(1.0 if s["default_picked"] else 0.0)
    return [
        {"day_id": day, "pick_rate": sum(v) / len(v), "taps": len(v)}
        for day, v in sorted(by_day.items())
    ]


def _ratings_by_day(conn: Any, user_id: str) -> list[dict[str, Any]]:
    """Per day: how many images the person rated off / ok / great (latest rating per image).
    The strip's pick is what they usually rate, so this is the direct answer to "are the images
    getting better?" — the share rated great should climb."""
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT DISTINCT ON (r.day_id, r.panel_id, r.url) r.day_id, r.rating
            FROM image_ratings r JOIN days d ON d.job_id = r.day_id
            WHERE d.user_id = %s AND d.kind = 'daily'
            ORDER BY r.day_id, r.panel_id, r.url, r.created_at DESC, r.id DESC
        )
        SELECT d.date,
               count(*) FILTER (WHERE l.rating = 0) AS off,
               count(*) FILTER (WHERE l.rating = 1) AS ok,
               count(*) FILTER (WHERE l.rating = 2) AS great
        FROM latest l JOIN days d ON d.job_id = l.day_id
        GROUP BY d.date ORDER BY d.date
        """,
        (user_id,),
    ).fetchall()
    return [
        {"day_id": r["date"].isoformat(), "off": r["off"], "ok": r["ok"], "great": r["great"]}
        for r in rows
    ]


def _latest_taps(conn: Any, user_id: str) -> list[dict[str, Any]]:
    """One tap per panel — the latest — as `{day_id, date, panel_id, chosen, others}`. A best-of-3
    tap wrote one `image_pairs` row per option passed on, sharing a `tap_id`; an older single-pair
    tap is a group of one (`COALESCE(tap_id, id)`)."""
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT DISTINCT ON (ip.day_id, ip.panel_id)
                   ip.day_id, ip.panel_id, COALESCE(ip.tap_id, ip.id::text) AS tid
            FROM image_pairs ip JOIN days d ON d.job_id = ip.day_id
            WHERE d.user_id = %s AND d.kind = 'daily' AND ip.source = 'ab'
            ORDER BY ip.day_id, ip.panel_id, ip.created_at DESC, ip.id DESC
        )
        SELECT l.day_id, d.date, l.panel_id, ip.chosen, ip.rejected
        FROM latest l
        JOIN days d ON d.job_id = l.day_id
        JOIN image_pairs ip ON ip.day_id = l.day_id AND ip.panel_id = l.panel_id
                           AND COALESCE(ip.tap_id, ip.id::text) = l.tid
        ORDER BY d.date, l.panel_id, ip.id
        """,
        (user_id,),
    ).fetchall()
    taps: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        tap = taps.setdefault(
            (row["day_id"], row["panel_id"]),
            {
                "day_id": row["day_id"],
                "date": row["date"],
                "panel_id": row["panel_id"],
                "chosen": row["chosen"],
                "others": [],
            },
        )
        if row["rejected"] not in tap["others"]:
            tap["others"].append(row["rejected"])
    return list(taps.values())


def _first_choice_agreement(conn: Any, user_id: str) -> list[dict[str, Any]]:
    """Per day: how often the person's pick was the app's own first choice among the options
    shown. The app's ranking is the `reward` it stored on each candidate when it chose; a tie
    counts 1/k. Only the latest tap per panel counts (a changed mind replaces the earlier tap)."""
    taps = _latest_taps(conn, user_id)
    if not taps:
        return []
    rewards = {
        (r["day_id"], r["url"]): r["reward"]
        for r in conn.execute(
            "SELECT day_id, url, (scores->>'reward')::float AS reward FROM candidates "
            "WHERE day_id = ANY(%s) AND scores ? 'reward'",
            (sorted({t["day_id"] for t in taps}),),
        ).fetchall()
    }
    by_day: dict[str, list[float]] = {}
    for tap in taps:
        shown = [tap["chosen"], *tap["others"]]
        values = [rewards.get((tap["day_id"], url)) for url in shown]
        if any(v is None for v in values):
            continue
        top = max(v for v in values if v is not None)
        winners = [i for i, v in enumerate(values) if v is not None and v >= top - 1e-9]
        by_day.setdefault(tap["date"].isoformat(), []).append(
            1.0 / len(winners) if 0 in winners else 0.0
        )
    return [
        {"day_id": day, "pick_rate": sum(v) / len(v), "taps": len(v)}
        for day, v in sorted(by_day.items())
    ]


@router.get("")
async def learning(user_id: str = Depends(current_user_id)) -> dict[str, Any]:
    with get_conn() as conn:
        pick_rate_series = _first_choice_agreement(conn, user_id)
        ratings_by_day = _ratings_by_day(conn, user_id)

        # "Taps" the model learns from: one per panel (the latest), A/B source only —
        # distinct from counting every `image_pairs` row, which would also include
        # regenerate rows and a panel tapped more than once.
        tap_count = fetch_one(
            conn,
            """
            SELECT count(DISTINCT (ip.day_id, ip.panel_id)) AS n FROM image_pairs ip
            JOIN days d ON d.job_id = ip.day_id
            WHERE d.user_id = %s AND d.kind = 'daily' AND ip.source = 'ab'
            """,
            (user_id,),
        )["n"]
        regeneration_count = fetch_one(
            conn,
            """
            SELECT count(*) AS n FROM image_pairs ip
            JOIN days d ON d.job_id = ip.day_id
            WHERE d.user_id = %s AND ip.source = 'regenerate'
            """,
            (user_id,),
        )["n"]
        caption_edit_count = fetch_one(
            conn,
            """
            SELECT count(*) AS n FROM caption_pairs cp
            JOIN days d ON d.job_id = cp.day_id WHERE d.user_id = %s
            """,
            (user_id,),
        )["n"]
        checkpoints = conn.execute(
            "SELECT id, kind, metrics, created_at FROM checkpoints ORDER BY created_at DESC"
        ).fetchall()

    return {
        "pick_rate_series": pick_rate_series,
        "ratings_by_day": ratings_by_day,
        "pairs": tap_count,  # kept for older clients; it now means distinct A/B taps
        "taps": tap_count,
        "regenerations": regeneration_count,
        "caption_edits": caption_edit_count,
        "checkpoints": [dict(c) for c in checkpoints],
        "next_run": _next_sunday_0200(),
        "preference": _preference(user_id),
    }
