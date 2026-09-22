# 0014. Memory: pgvector-backed store (Phase 7, task 7.3)

Status: Accepted (2026-09-15)

## Context

Task 7.3 asks for a real `memory.py` writing beats/people/places with
`bge-small-en-v1.5` embeddings over pgvector, plus `recent_people`,
`known_vocab`, `place_ref`, and `search` (returning beats + panel URLs).
Building this against the real local Postgres (ADR 0009) surfaced one
real ordering problem, one real embedding-API detail, and two real bugs
caught by insisting on a live end-to-end test rather than mocking the
database away.

## Decision — schema bootstrap now, Alembic later

Task 8.1 (Alembic migrations for every table in CLAUDE.md §3) doesn't
exist yet, but 7.3 needs real tables today. `app/db.py`'s `ensure_schema`
creates a small, deliberately `memory_`-prefixed table set
(`memory_beats`, `memory_people`, `memory_places`) idempotently
(`CREATE TABLE IF NOT EXISTS`) — scoped to exactly what `memory.py`
needs, not the full table set from §3. This is a temporary bootstrap:
task 8.1 should absorb or formalize these, not leave two schema-creation
paths running forever.

## Decision — BGE-small: CLS pooling, asymmetric query/passage embedding

Verified live against the model's own card before writing any code:
`bge-small-en-v1.5` uses **CLS-token** pooling (the first hidden state),
not mean pooling — the more common convention for sentence-embedding
models, and an easy wrong guess. Output is 384-dim, L2-normalized. The
model's retrieval convention is asymmetric: a short *query* gets a fixed
instruction prefix ("Represent this sentence for searching relevant
passages: "); indexed *passages* (a beat's text) get none. Mixing these
up doesn't error, it silently degrades ranking — so `app/embeddings.py`
exposes `embed_query`/`embed_passage` as two distinctly named functions,
not one function with a boolean easy to get backwards at a call site.

## Decision — `Storage`-style Local/Postgres split, not a replacement

`InMemoryMemoryStore` (task 2.2's interim store) stays as the default
for tests that don't need a database — most of Phases 2-6's existing
tests. `PostgresMemoryStore` (task 7.3) is the new real backend, added
alongside it, matching `packages/storage`'s own Local/R2 split. The
`MemoryStore` Protocol gained `recent_people`, `search`, and
`set_place_ref` (promoted from an `InMemoryMemoryStore`-only test
helper to a first-class method now that a real backend exists to
persist it) — implemented in both backends, so nothing that only needs
*a* `MemoryStore` requires a real database to test.

## Real bug 1 — `.env` silently not found outside repo-root cwd

Found running task 7.3's first live Postgres test: it skipped instead of
running, because `os.environ.get("DATABASE_URL")` was empty. Root cause
was deeper than my own new code — `services/{api,worker,scheduler}/app/
config.py`'s `Settings` all used a relative `env_file=".env"`, which
only resolves when the process's cwd is the repo root. `make dev`'s
Procfile-driven processes happen to satisfy this; `scripts/test-all.sh`'s
own documented convention (`cd services/worker && python -m pytest
../../tests/unit/llm`, needed for `app.*` imports to resolve) does not —
so `.env` was silently never found in that context. Latent until this
task needed a real `.env`-sourced variable inside a worker-cwd test
process for the first time; no existing test exercised `get_settings()`
directly, so nothing had caught it. Fixed in all three `config.py` files
with `find_dotenv(usecwd=True)`, which walks up from cwd to find the
repo-root `.env` regardless of where the process started; `app/db.py`
now goes through `get_settings().database_url` rather than raw
`os.environ`, so it inherits the same correct resolution.

## Real bug 2 — `SET search_path` hides the `vector` type

The first real write against the dedicated test schema failed with
`psycopg.ProgrammingError: vector type not found in the database`, even
though `\dx` confirms the extension is genuinely installed. Cause:
`SET search_path TO "comiccanvas_test"` *replaces* the path rather than
extending it, hiding `public` — where `CREATE EXTENSION vector` actually
put the `vector` type — from name resolution, so `pgvector.psycopg.
register_vector`'s type lookup came back empty. Fixed by setting
`search_path TO "{schema}", public` — the dedicated schema still takes
precedence for `memory.py`'s own tables, while the extension's type
stays visible.

## Verification

Real, not mocked, end to end: `bge-small-en-v1.5`'s real embedding
behavior was checked live in a scratch script before writing
`embeddings.py` (384-dim, unit-normalized, a real "gym" query scoring a
gym passage at 0.68 cosine similarity vs. 0.42 for an unrelated one).
19 new tests: `tests/unit/llm/test_embeddings.py` (4, mocked
tokenizer/model — CLS-pooling extraction, query-prefix behavior,
truncation), `tests/unit/llm/test_memory.py` (6, `InMemoryMemoryStore`'s
new methods), `tests/unit/llm/test_memory_postgres.py` (9, real local
Postgres in a dedicated `comiccanvas_test` schema, real BGE model,
`enable_socket` for the real TCP connection — includes task 7.3's
literal accept line, a "gym" query correctly returning the Sep 3 beat
first over cafe/kitchen/transit beats on Sep 1/2/4). `ruff`/
`mypy --strict` clean; full suite (447 tests) green via both direct
pytest and the real `scripts/test-all.sh` entrypoint.

## Consequences

- `app/config.py`'s `find_dotenv` fix applies to all three services, not
  just the worker — any future test that instantiates `get_settings()`
  from a non-root cwd now resolves `.env` correctly instead of silently
  failing validation or reading stale defaults.
- `known_vocab`'s real source (task 2.6/8.3's "known_vocab (top 40
  entities)") is an aggregation of known people, known places, and
  distinct beat objects — not separately specified in CLAUDE.md, decided
  here since 7.3 names it as a `PostgresMemoryStore` method.
- `top_rated_scripts` on `PostgresMemoryStore` returns `[]` — no
  `thumbs`/pick-data tables exist before Phase 8/9. Honest emptiness,
  not a fabricated ranking; revisit once that data exists.
- `search`'s `panel_urls` come from whatever `write_beats(...,
  panel_urls=...)` was given at write time — no `panels` table exists
  yet (task 8.1) to join against instead.
