# API reference

Generated from the real FastAPI app (`services/api/app/main.py`) via
`app.openapi()` (task 8.13). Regenerate after any router change:
`uv run python scripts/gen_api_docs.py` from the repo root.

The raw schema is committed at `docs/openapi.json`; this file is a
human-readable index into it, grouped by router.

## account

- `GET /account/export` — Export Account
- `DELETE /account` — Delete Account

## cast

- `GET /cast` — List Cast
- `POST /cast` — Create Person
- `PATCH /cast/{person_id}` — Update Person
- `DELETE /cast/{person_id}` — Revoke Person
- `POST /cast/{person_id}/photos/presign` — Presign Photo
- `POST /cast/{person_id}/process` — Process Person
- `POST /cast/{person_id}/train` — Train Person
- `POST /cast/{person_id}/master` — Pick Master
- `POST /cast/invite` — Create Invite

## days

- `POST /days` — Create Day
- `GET /jobs/{job_id}/events` — Job Events
- `GET /days/{date}` — Get Day

## feedback

- `POST /feedback/pair` — Feedback Pair
- `POST /feedback/pick` — Feedback Pick
- `POST /feedback/rating` — Feedback Rating
- `POST /feedback/caption` — Feedback Caption
- `POST /feedback/thumb` — Feedback Thumb
- `POST /days/{date}/panels/{panel_id}/regenerate` — Regenerate Panel

## learning

- `GET /learning` — Learning

## library

- `GET /library` — Library
- `GET /search` — Search

## media

- `PUT /media/upload` — Upload
- `GET /media/{key}` — Serve

## settings

- `GET /settings` — Get User Settings
- `PUT /settings` — Put User Settings

## untagged

- `GET /health` — Health

## weekly

- `POST /weekly/{iso_week}/generate` — Generate Weekly
- `GET /weekly/{iso_week}.pdf` — Get Weekly Pdf
- `GET /weekly/{iso_week}` — Get Weekly

