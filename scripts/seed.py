"""`make seed` — task 8.1's fixtures: one user, 7 days, cast of 3.

Real inserts against `DATABASE_URL` (ADR 0009/0017), through the tables
`migrations/versions/af5692fa786e_initial_schema.py` creates. Idempotent:
deletes any prior run's rows by their fixed seed ids first (`ON DELETE
CASCADE` from `users`/`people` handles the rest), so `make seed` is safe
to re-run against the same dev database.

Each day gets one beat/panel/candidate too (not just a bare row) — the
minimum needed for task 8.5's `GET /days/{date}` and 8.7's library grid
to have something real to return once those endpoints exist.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import psycopg
from dotenv import find_dotenv, load_dotenv

USER_ID = "seed-user"
PEOPLE = [("seed-person-1", "Sam"), ("seed-person-2", "Alex"), ("seed-person-3", "Jamie")]
PLACES = ["gym", "kitchen", "desk", "outdoors", "cafe", "bedroom", "transit"]


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL not set — see README.md's Setup section")

    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (USER_ID,))

        cur.execute(
            "INSERT INTO users (id, email) VALUES (%s, %s)",
            (USER_ID, "seed@comiccanvas.example"),
        )

        for person_id, name in PEOPLE:
            cur.execute(
                "INSERT INTO people (id, user_id, name) VALUES (%s, %s, %s)",
                (person_id, USER_ID, name),
            )

        start = date.today() - timedelta(days=6)
        for i in range(7):
            day = start + timedelta(days=i)
            job_id = f"seed-job-{i}"
            place = PLACES[i % len(PLACES)]
            cast = [PEOPLE[i % 3][0]]

            cur.execute(
                "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'daily', 'done')",
                (job_id, USER_ID),
            )
            cur.execute(
                "INSERT INTO days (job_id, user_id, date, source, text, mood) "
                "VALUES (%s, %s, %s, 'text', %s, 'content')",
                (job_id, USER_ID, day, f"Seed day {i + 1}: a trip to the {place}."),
            )
            cur.execute(
                "INSERT INTO beats (day_id, beat_id, time, place, event, emotion, "
                "people, importance, humor) VALUES "
                "(%s, 'b1', 'evening', %s, %s, 'content', %s, 0.6, 0.3)",
                (job_id, place, f"went to the {place}", cast),
            )
            cur.execute(
                "INSERT INTO panels (day_id, panel_id, beat_id, place, time_of_day, "
                "expression, action, framing, caption_a, cast_ids) VALUES "
                "(%s, 1, 'b1', %s, 'evening', 'content', %s, 'wide', %s, %s)",
                (job_id, place, f"walking into the {place}", f"Day {i + 1}", cast),
            )
            cur.execute(
                "INSERT INTO candidates (id, day_id, panel_id, url, seed, chosen) "
                "VALUES (%s, %s, 1, %s, %s, true)",
                (f"seed-candidate-{i}", job_id, f"/media/seed/day{i + 1}/panel1.png", 1000 + i),
            )

        conn.commit()

    print(f"Seeded: 1 user ({USER_ID}), 7 days, cast of {len(PEOPLE)}.")


if __name__ == "__main__":
    main()
