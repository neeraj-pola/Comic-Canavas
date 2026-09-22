# 0019. API surface and worker jobs (Phase 8, tasks 8.2–8.14)

Status: Accepted (2026-09-16)

## Context

Phase 8 built the real FastAPI surface (`services/api`) and the real arq
jobs that back it (`services/api/app/worker.py`), wiring the pipeline
built in Phases 1–7 to a real database and a real, callable API for the
first time. This ADR records the real, non-obvious decisions and bugs
found doing that — not a restatement of every route (see `docs/api.md`,
generated from the real running app's own OpenAPI schema).

## Decision — `run_daily_job`/`run_weekly_job` live in the worker, the API only enqueues

`POST /days` and `POST /weekly/{iso_week}/generate` create a `jobs` row
and `arq_pool.enqueue_job(...)`; the actual `DayGraph`/`build_weekly_recap`
run happens in `services/worker/app/worker.py`, never in the API process.
`DayGraph.run_streaming` (new: `astream(..., stream_mode="updates")`,
verified live against a throwaway two-node graph before trusting it, same
discipline ADR 0015 established) lets the worker append one `jobs.events`
entry per completed node, which `GET /jobs/{id}/events` polls and streams
as SSE — hand-rolled (`StreamingResponse` + `text/event-stream`), not a
client library, since the two processes only share Postgres and Redis.

## Decision — `app.persist` is the missing write path into task 8.1's tables

Nothing before this ever wrote a finished `DayState` into
`days`/`beats`/`panels`/`candidates` — task 7.3's memory node writes to
the separate `memory_beats` (pgvector search), and `compose_day` writes
image bytes to `Storage`, neither of which `GET /days/{date}` can query.
`persist_day_state` (idempotent, `ON CONFLICT ... DO UPDATE`) is that
write path, called once after the graph finishes. The weekly recap reuses
it unchanged: its top-6 beats are copied into a **fresh** `beats` row set
scoped to the weekly `job_id` (not the original days'), so `panels`'
`(day_id, beat_id)` foreign key is satisfiable without a new
`weekly_beats`/`weekly_panels` table CLAUDE.md §3 never names.

## Real bug — LangGraph's own `checkpoints` table collided with task 8.1's

`open_day_graph`'s Postgres checkpointer used to default to `public` (no
`schema` override) for real usage. The very first real run of
`run_daily_job` against the real dev database failed with
`psycopg.errors.UndefinedColumn: column "thread_id" does not exist` —
`AsyncPostgresSaver.setup()` hardcodes a table named `checkpoints` for
its own graph-state bookkeeping, which collided with task 8.1's real
`checkpoints` table (Phase 9's model-checkpoint registry, CLAUDE.md §3),
already present with a completely different, incompatible shape.
`setup()` found the wrong table and tried to migrate it in place.

Fixed by giving LangGraph's own tables a permanently dedicated schema
(`langgraph`, `open_day_graph`'s new default — real usage no longer ever
passes `schema=None`); the three orphaned tables (`checkpoint_blobs`,
`checkpoint_migrations`, `checkpoint_writes`) left behind by the failed
`public` attempt were dropped by hand. Tests needing full isolation still
pass an explicit `schema=` (task 7.6's own convention), kept deliberately
separate from whatever schema holds the business-data tables for that
test — mixing the two back together would reproduce the exact bug.

## Real bug — `candidates.seed` overflowed `INT`

`seed_for_panel` (task 3.7) derives seeds from 8 hex digits of a sha256
hash — up to 0xFFFFFFFF (4,294,967,295) — but Postgres `INT` is a
**signed** 32-bit integer (max 2,147,483,647). The first real candidate
insert through `persist_candidates` overflowed with
`NumericValueOutOfRange`. Fixed with a new migration
(`fa6f0a01842d_widen_candidates_seed_to_bigint.py`) widening the column
to `BIGINT` — no application code needed to change.

## Real bug — `/weekly/{iso_week}.pdf` route-ordering

`@router.get("/{iso_week}")` was registered before
`@router.get("/{iso_week}.pdf")`. Starlette matches routes in
registration order, and a bare `{iso_week}` path parameter has no `.`
restriction, so it silently swallowed `"2026-W38.pdf"` whole as
`iso_week` itself, 422ing on the (correctly rejecting) ISO-week regex.
Fixed by registering the more specific `.pdf` route first — found live
by the first real PDF-download test, not by inspection.

## Real constraint — `ml.identity.lookcard`'s sys.path bootstrap doesn't survive a second `app` package

`ml/identity/lookcard.py` (task 5.4) inserts `services/worker` onto
`sys.path` so its one LLM call can `import app.llm` — a trick that only
ever ran from a process where nothing else had already claimed the name
`app` (e.g. `ml/evals/*.py` run standalone). The first time
`services/api`'s own `cast.py` (living inside API's own `app.routers`)
imported it, `app` was already bound to **API's** package in
`sys.modules`, so `import app.llm` raised `ModuleNotFoundError: No
module named 'app.llm'` — the two services' both using a private
top-level `app` namespace (CLAUDE.md §2's own deliberate layout) directly
conflicts with that bootstrap once both are live in one process.

Resolved by scope, not by a new import trick: `POST /cast/{id}/process`
only calls `ml.identity.faces`/`ml.identity.embedding` (genuinely
`app.*`-free — real face detection/embedding, free, synchronous, safe in
the API). Look-card generation (the one step that needs `app.llm`) moved
into the worker's `train_character_job`, run once before `design_master`
needs it. `services/api/pyproject.toml` and `services/worker/pyproject.toml`
both now declare `comiccanvas-ml` as a real workspace dependency (it's a
properly installed package, `ml.*`, unlike the private `app.*` namespaces)
rather than relying on incidental shared-venv installation.

## Decision — `/search` is lexical, not semantic

Task 7.3's `PostgresMemoryStore.search()` is real pgvector cosine search
over `bge-small-en-v1.5` embeddings, but it lives in `services/worker`'s
own `app` package. Pulling `torch`/`transformers` into the lightweight
API service just for one read endpoint was judged a disproportionate
dependency; `GET /search` instead runs `ILIKE` over the real relational
`beats` table (task 8.1) — a real, sensible service boundary, documented
in `app/routers/library.py`'s own module docstring, not a silently worse
stand-in for the same thing.

## Decision — task 8.13 scoped to `docs/api.md`, TS types deferred to Phase 10

`scripts/gen_api_docs.py` regenerates `docs/openapi.json` and
`docs/api.md` from the real running app's own `app.openapi()` — real,
automatable, done. `pnpm gen:types` (the TS half of both task 8.13 and
task 10.15) has nothing to generate into yet: `apps/web` doesn't exist
until Phase 10. Not a gap silently carried forward — task 10.15 already
names this as its own accept line.

## Verification

Real, not just unit-tested: a manual smoke test (`uv run uvicorn`/`arq`
against the real dev Postgres, `IMAGE_PROVIDER=mock`) is what surfaced
the checkpoint-schema-collision and seed-overflow bugs in the first
place, before any automated test existed to catch them. `tests/unit/llm/
test_worker_jobs.py` (2 tests) now covers `run_daily_job` end to end for
real — real Postgres, isolated `comiccanvas_test_worker_jobs` schema,
real migration, asserting `jobs.status='done'`, the exact 8-node event
order, and real `panels`/`candidates` rows. `tests/unit/api/` (35 tests
across 8 router files) covers every route's real read/write behavior
against a real, isolated `comiccanvas_test_api` schema — including a
real Playwright-rendered PDF (`%PDF` magic bytes checked, not mocked).
`tests/unit/scheduler/test_cron.py` (10 tests) covers all four cron
entrypoints with frozen time (an explicit `now: datetime` parameter, no
time-mocking library). `tests/e2e/test_backend_flow.py` is Gate 8's own
literal accept line: sign-in → onboarding → day → feedback → weekly PDF
→ export → delete, one continuous test, real Postgres, real PDF bytes.
`ruff`/`mypy --strict` clean across every new file; full suite (516
tests) green via `make test`.

## Consequences

- Any future service that needs both a private `app.*` package (its own
  FastAPI/worker code) AND something from `ml.identity.lookcard` (or any
  other `ml/` module with the same sys.path bootstrap) will hit the same
  conflict this ADR documents — the fix pattern is "move the `app.llm`-
  dependent step to the process that legitimately owns `app.worker`,"
  not a new sys.path trick.
- `open_day_graph`'s checkpointer schema (`langgraph`) is now permanent,
  real infrastructure — a future migration must never create a
  `checkpoints`/`checkpoint_writes`/`checkpoint_blobs`/
  `checkpoint_migrations` table in `public` for an unrelated purpose.
- `identity_models` is resolved by "the first `identity_models` row for
  the user's `people`" (Gate 5's single-character scope) — task 5.17
  (recurring characters) will need a real `people.is_owner`-style column
  before more than one character per user is meaningful here.
