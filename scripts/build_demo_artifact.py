"""One-off script: builds artifact/ 's static demo dataset from the real
local database and real generated images. Not part of the app itself —
run once (re-run any time to refresh) to regenerate artifact/public/demo/*
and artifact/lib/demo-data.json.

Usage: uv run --project services/worker python scripts/build_demo_artifact.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import find_dotenv, load_dotenv
from psycopg.rows import dict_row
from PIL import Image

load_dotenv(find_dotenv(usecwd=True))

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / ".data"
ARTIFACT = REPO_ROOT / "artifact"
PUBLIC_DEMO = ARTIFACT / "public" / "demo"
USER_ID = "user_3JeXeYKZmkweXnCQZLXw89Wgo3f"
MEDIA_PREFIX = "http://localhost:8000/media/"

# Resized so the demo repo stays small — same real images, web-sized.
STRIP_MAX = 1200
PANEL_MAX = 900
JPEG_QUALITY = 78


def local_path_for(url: str) -> Path | None:
    if not url or not url.startswith(MEDIA_PREFIX):
        return None
    return DATA_DIR / url[len(MEDIA_PREFIX) :]


def copy_resized(url: str | None, max_edge: int) -> str | None:
    """Copies+resizes a real generated image into public/demo, returns its
    new root-relative URL (None if the source file doesn't exist)."""
    if not url:
        return None
    src = local_path_for(url)
    if src is None or not src.exists():
        return None
    rel = url[len(MEDIA_PREFIX) :]
    dest_rel = Path(rel).with_suffix(".jpg")
    dest = PUBLIC_DEMO / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_edge, max_edge), Image.LANCZOS)
            im.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return f"/demo/{dest_rel.as_posix()}"


def fetch_all(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] = ()) -> list[dict]:
    return conn.execute(sql, params).fetchall()  # type: ignore[return-value]


def fetch_one(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] = ()) -> dict | None:
    return conn.execute(sql, params).fetchone()  # type: ignore[return-value]


def build_day(conn: psycopg.Connection[Any], row: dict) -> dict[str, Any]:
    job_id = row["job_id"]
    date = row["date"].isoformat()
    max_edge = STRIP_MAX

    beats = fetch_all(conn, "SELECT * FROM beats WHERE day_id = %s ORDER BY beat_id", (job_id,))
    panels = fetch_all(
        conn, "SELECT * FROM panels WHERE day_id = %s ORDER BY panel_id", (job_id,)
    )
    candidates = fetch_all(
        conn,
        "SELECT id, panel_id, url, seed, scores, face_box, chosen, rejected_reason, prompt_id "
        "FROM candidates WHERE day_id = %s ORDER BY panel_id, created_at",
        (job_id,),
    )
    prompts = fetch_all(
        conn,
        "SELECT id, panel_id, generator, character_clause, environment_clause, positive, "
        "negative, seed, guidance, steps FROM image_prompts WHERE day_id = %s ORDER BY panel_id",
        (job_id,),
    )
    picks = fetch_all(
        conn,
        "SELECT DISTINCT ON (panel_id) panel_id, chosen FROM image_pairs "
        "WHERE day_id = %s AND source = 'ab' ORDER BY panel_id, created_at DESC",
        (job_id,),
    )
    ratings = fetch_all(
        conn,
        "SELECT DISTINCT ON (panel_id, url) panel_id, url, rating FROM image_ratings "
        "WHERE day_id = %s ORDER BY panel_id, url, created_at DESC",
        (job_id,),
    )

    # Resize+copy every candidate image, remap urls in place.
    url_map: dict[str, str] = {}
    for c in candidates:
        new_url = copy_resized(c["url"], PANEL_MAX)
        if new_url:
            url_map[c["url"]] = new_url
            c["url"] = new_url
        c.pop("face_box_raw", None)
        if c["face_box"] is not None:
            c["face_box"] = list(c["face_box"])

    strip_url = copy_resized(row["strip_url"], max_edge)
    story_url = copy_resized(row["story_url"], max_edge)

    picks_map = {str(p["panel_id"]): url_map.get(p["chosen"], p["chosen"]) for p in picks}
    ratings_map = {url_map.get(r["url"], r["url"]): r["rating"] for r in ratings}

    return {
        "date": date,
        "source": row["source"],
        "text": row["text"],
        "transcript": row["transcript"],
        "mood": row["mood"],
        "quiet_day": row["quiet_day"],
        "strip_url": strip_url,
        "story_url": story_url,
        "layout": row["layout"],
        "cost_usd": float(row["cost_usd"]),
        "versions": row["versions"] or {},
        "picks": picks_map,
        "ratings": ratings_map,
        "beats": [
            {
                "id": b["beat_id"],
                "time": b["time"],
                "place": b["place"],
                "place_detail": b["place_detail"],
                "event": b["event"],
                "emotion": b["emotion"],
                "people": b["people"],
                "objects": b["objects"],
                "importance": b["importance"],
                "humor": b["humor"],
                "quote": b["quote"],
            }
            for b in beats
        ],
        "panels": [
            {
                "id": p["panel_id"],
                "beat_id": p["beat_id"],
                "place": p["place"],
                "time_of_day": p["time_of_day"],
                "expression": p["expression"],
                "action": p["action"],
                "framing": p["framing"],
                "caption_a": p["caption_a"],
                "caption_b": p["caption_b"],
                "bubble": p["bubble"],
                "cast": p["cast_ids"],
            }
            for p in panels
        ],
        "candidates": [
            {
                "id": c["id"],
                "panel_id": c["panel_id"],
                "url": c["url"],
                "seed": c["seed"],
                "scores": c["scores"],
                "face_box": c["face_box"],
                "chosen": c["chosen"],
                "rejected_reason": c["rejected_reason"],
                "prompt_id": c["prompt_id"],
            }
            for c in candidates
        ],
        "prompts": [
            {
                "id": pr["id"],
                "panel_id": pr["panel_id"],
                "generator": pr["generator"],
                "character_clause": pr["character_clause"],
                "environment_clause": pr["environment_clause"],
                "positive": pr["positive"],
                "negative": pr["negative"],
                "seed": pr["seed"],
                "guidance": pr["guidance"],
                "steps": pr["steps"],
            }
            for pr in prompts
        ],
    }


def build_weekly(
    conn: psycopg.Connection[Any], row: dict, week_days: list[dict]
) -> dict[str, Any]:
    job_id = row["job_id"]
    panels = fetch_all(
        conn, "SELECT * FROM panels WHERE day_id = %s ORDER BY panel_id", (job_id,)
    )
    candidates = fetch_all(
        conn,
        "SELECT panel_id, url FROM candidates WHERE day_id = %s AND chosen ORDER BY panel_id",
        (job_id,),
    )
    by_panel = {c["panel_id"]: c["url"] for c in candidates}
    weekly_panels = []
    for p in panels:
        url = copy_resized(by_panel.get(p["panel_id"]), PANEL_MAX)
        weekly_panels.append({"id": p["panel_id"], "caption": p["caption_a"], "url": url})

    day_count = fetch_one(
        conn,
        "SELECT count(*) AS n FROM days WHERE user_id = %s AND kind = 'daily' "
        "AND strip_url IS NOT NULL",
        (USER_ID,),
    )
    pair_count = fetch_one(
        conn,
        "SELECT count(*) AS n FROM image_pairs ip JOIN days d ON d.job_id = ip.day_id "
        "WHERE d.user_id = %s AND ip.source = 'ab'",
        (USER_ID,),
    )

    return {
        "week_label": "2026-W37",
        "date_range": "Sep 7 to Sep 13, 2026",
        "user_label": "you",
        "mood": row["mood"],
        "quiet_day": row["quiet_day"],
        "strip_url": copy_resized(row["strip_url"], STRIP_MAX),
        "story_url": copy_resized(row["story_url"], STRIP_MAX),
        "panels": weekly_panels,
        "mood_series": [d["mood"] for d in week_days],
        "stats": {
            "days": day_count["n"] if day_count else 0,
            "panels": (day_count["n"] if day_count else 0) * 4,
            "pairs": pair_count["n"] if pair_count else 0,
            "first_pick_pct": 0.42,
        },
        "cast": [{"name": "you"}],
        "training_run": None,
    }


def build_person(conn: psycopg.Connection[Any]) -> dict[str, Any] | None:
    person = fetch_one(conn, "SELECT * FROM people WHERE user_id = %s LIMIT 1", (USER_ID,))
    if not person:
        return None
    identity = fetch_one(
        conn, "SELECT * FROM identity_models WHERE person_id = %s", (person["id"],)
    )
    master_path = None
    master_candidates: list[dict[str, Any]] = []
    identity_out = None
    if identity:
        master_path = copy_resized(identity["master_path"], PANEL_MAX)
        raw_candidates = identity["master_candidates"] or []
        for mc in raw_candidates:
            url = copy_resized(mc.get("url"), PANEL_MAX)
            if url:
                master_candidates.append({**mc, "url": url})
        identity_out = {
            "person_id": identity["person_id"],
            "source_photo_count": identity["source_photo_count"] or 0,
            "mean_self_cosine": identity["mean_self_cosine"],
            "look_card": identity["look_card"],
            "master_path": master_path,
            "generation_path": identity["generation_path"],
            "master_candidates": master_candidates or None,
        }
    return {
        "id": person["id"],
        # Public demo: generic label, not the real account's name, matching
        # this app's own single-user "You" convention (NEXT_PUBLIC_DEV_USER_NAME).
        "name": "You",
        "gender_term": person["gender_term"],
        "age": person["age"],
        "consent_at": person["consent_at"].isoformat() if person["consent_at"] else None,
        "revoked_at": None,
        "created_at": person["created_at"].isoformat(),
        "identity": identity_out,
        "training": None,
    }


def build_settings(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    row = fetch_one(conn, "SELECT * FROM settings WHERE user_id = %s", (USER_ID,))
    # The demo represents the real, current default product behavior, not
    # this dev account's own leftover test values (e.g. generator="mock").
    defaults = {
        "style": None,
        "humor": row["humor"] if row else 5,
        "caption_length": row["caption_length"] if row and row["caption_length"] else "short",
        "sensitive_mode": row["sensitive_mode"] if row else True,
        "reminder_time": row["reminder_time"] if row and row["reminder_time"] else "20:30",
        "channel": row["channel"] if row and row["channel"] else "push",
        "generator": "flux_kontext",
        "detail_level": row["detail_level"] if row and row["detail_level"] else "rich",
        "personalise": row["personalise"] if row else True,
    }
    return defaults


def build_learning(conn: psycopg.Connection[Any], url_map: dict[str, str]) -> dict[str, Any]:
    model = fetch_one(conn, "SELECT * FROM preference_models WHERE user_id = %s", (USER_ID,))
    snapshots_raw = fetch_all(
        conn,
        "SELECT s.tap_index AS tap, d.date, s.panel_id, s.chosen_url, s.rejected_url, "
        "s.rejected_urls, s.default_picked, s.prob, s.correct, s.hand_correct, s.mu, s.sd, "
        "s.delta, s.axis FROM preference_snapshots s LEFT JOIN days d ON d.job_id = s.day_id "
        "WHERE s.user_id = %s ORDER BY s.tap_index",
        (USER_ID,),
    )
    explore = fetch_all(
        conn,
        "SELECT d.date, count(*) AS panels, "
        "count(*) FILTER (WHERE (c.scores->>'explored')::float >= 0.5) AS explored "
        "FROM candidates c JOIN days d ON d.job_id = c.day_id "
        "WHERE d.user_id = %s AND d.kind = 'daily' AND c.chosen AND c.scores ? 'explored' "
        "GROUP BY d.date ORDER BY d.date",
        (USER_ID,),
    )
    ratings_by_day = fetch_all(
        conn,
        """
        WITH latest AS (
            SELECT DISTINCT ON (r.day_id, r.panel_id, r.url) r.day_id, r.rating
            FROM image_ratings r JOIN days d ON d.job_id = r.day_id
            WHERE d.user_id = %s AND d.kind = 'daily'
            ORDER BY r.day_id, r.panel_id, r.url, r.created_at DESC, r.id DESC
        )
        SELECT d.date, count(*) FILTER (WHERE l.rating = 0) AS off,
               count(*) FILTER (WHERE l.rating = 1) AS ok,
               count(*) FILTER (WHERE l.rating = 2) AS great
        FROM latest l JOIN days d ON d.job_id = l.day_id GROUP BY d.date ORDER BY d.date
        """,
        (USER_ID,),
    )
    checkpoints = fetch_all(
        conn, "SELECT id, kind, metrics, created_at FROM checkpoints ORDER BY created_at DESC"
    )
    tap_count = fetch_one(
        conn,
        "SELECT count(DISTINCT (ip.day_id, ip.panel_id)) AS n FROM image_pairs ip "
        "JOIN days d ON d.job_id = ip.day_id "
        "WHERE d.user_id = %s AND d.kind = 'daily' AND ip.source = 'ab'",
        (USER_ID,),
    )
    caption_count = fetch_one(
        conn,
        "SELECT count(*) AS n FROM caption_pairs cp JOIN days d ON d.job_id = cp.day_id "
        "WHERE d.user_id = %s",
        (USER_ID,),
    )

    def remap(u: str | None) -> str:
        if not u:
            return u
        return url_map.get(u, u)

    snapshots = []
    for s in snapshots_raw:
        snapshots.append(
            {
                "tap": s["tap"],
                "date": s["date"].isoformat() if s["date"] else None,
                "panel_id": s["panel_id"],
                "chosen_url": remap(s["chosen_url"]),
                "rejected_url": remap(s["rejected_url"]),
                "rejected_urls": [remap(u) for u in (s["rejected_urls"] or [])],
                "default_picked": s["default_picked"],
                "prob": s["prob"],
                "correct": s["correct"],
                "hand_correct": s["hand_correct"],
                "mu": s["mu"],
                "sd": s["sd"],
                "delta": s["delta"],
                "axis": s["axis"],
            }
        )

    preference = {
        "features": model["features"] if model else [],
        "n_taps": model["n_taps"] if model else 0,
        "active": bool(model["active"]) if model else False,
        "lean": model["lean"] if model else {},
        "z": model["z"] if model else {},
        "accuracy": model["accuracy"] if model else {},
        "knobs": model["knobs"] if model else {},
        "notes": model["notes"] if model else {},
        "default_pick_by_day": _default_pick_by_day(snapshots),
        "snapshots": snapshots,
        "explore_by_day": [
            {"date": e["date"].isoformat(), "panels": e["panels"], "explored": e["explored"]}
            for e in explore
        ],
    }
    return {
        "preference": preference,
        "ratings_by_day": [
            {"day_id": r["date"].isoformat(), "off": r["off"], "ok": r["ok"], "great": r["great"]}
            for r in ratings_by_day
        ],
        "pick_rate_series": _first_choice_by_day(snapshots),
        "pairs": tap_count["n"] if tap_count else 0,
        "taps": tap_count["n"] if tap_count else 0,
        "regenerations": 0,
        "caption_edits": caption_count["n"] if caption_count else 0,
        "checkpoints": [
            {
                "id": c["id"],
                "kind": c["kind"],
                "metrics": c["metrics"] or {},
                "created_at": c["created_at"].isoformat(),
            }
            for c in checkpoints
        ],
        "next_run": "2026-09-28T02:00:00",
    }


def _default_pick_by_day(snapshots: list[dict]) -> list[dict[str, Any]]:
    by_day: dict[str, list[float]] = {}
    for s in snapshots:
        if s["default_picked"] is None or s["date"] is None:
            continue
        by_day.setdefault(s["date"], []).append(1.0 if s["default_picked"] else 0.0)
    return [
        {"day_id": day, "pick_rate": sum(v) / len(v), "taps": len(v)}
        for day, v in sorted(by_day.items())
    ]


def _first_choice_by_day(snapshots: list[dict]) -> list[dict[str, Any]]:
    by_day: dict[str, list[float]] = {}
    for s in snapshots:
        if s["date"] is None:
            continue
        by_day.setdefault(s["date"], []).append(s["correct"])
    return [
        {"day_id": day, "pick_rate": sum(v) / len(v), "taps": len(v)}
        for day, v in sorted(by_day.items())
    ]


def main() -> None:
    database_url = os.environ["DATABASE_URL"]
    PUBLIC_DEMO.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        day_rows = fetch_all(
            conn,
            "SELECT * FROM days WHERE user_id = %s AND kind = 'daily' AND strip_url IS NOT NULL "
            "ORDER BY date",
            (USER_ID,),
        )
        weekly_rows = fetch_all(
            conn,
            "SELECT * FROM days WHERE user_id = %s AND kind = 'weekly' AND strip_url IS NOT NULL "
            "ORDER BY date",
            (USER_ID,),
        )

        days = [build_day(conn, r) for r in day_rows]

        # Build a global url_map (original -> demo) from every candidate/strip
        # touched above, for remapping learning-page snapshot urls too.
        url_map: dict[str, str] = {}
        for d, r in zip(days, day_rows, strict=True):
            if d["strip_url"]:
                url_map[r["strip_url"]] = d["strip_url"]
            if d["story_url"]:
                url_map[r["story_url"]] = d["story_url"]
        cand_rows = fetch_all(
            conn,
            "SELECT c.url AS raw_url FROM candidates c JOIN days d ON d.job_id = c.day_id "
            "WHERE d.user_id = %s",
            (USER_ID,),
        )
        for row in cand_rows:
            raw = row["raw_url"]
            mapped = copy_resized(raw, PANEL_MAX)
            if mapped:
                url_map[raw] = mapped

        week_days = [d for d in days if d["date"] <= "2026-09-13"]
        weekly = build_weekly(conn, weekly_rows[0], week_days) if weekly_rows else None
        person = build_person(conn)
        settings = build_settings(conn)
        learning = build_learning(conn, url_map)

    library_days = [
        {
            "date": d["date"],
            "mood": d["mood"],
            "quiet_day": d["quiet_day"],
            "strip_url": d["strip_url"],
            "thumbs": [],
        }
        for d in days
    ]

    output = {
        "days": {d["date"]: d for d in days},
        "library_days": library_days,
        "weekly": {"2026-W37": weekly} if weekly else {},
        "person": person,
        "settings": settings,
        "learning": learning,
    }

    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / "lib").mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT / "lib" / "demo-data.json"
    out_path.write_text(json.dumps(output, indent=None, default=str))
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"wrote {out_path} ({size_mb:.1f} MB)")
    print(f"days: {len(days)}, weekly: {bool(weekly)}, person: {bool(person)}")


if __name__ == "__main__":
    main()
