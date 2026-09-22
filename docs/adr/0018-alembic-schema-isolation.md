# 0018. Alembic migrations, raw SQL, and test-schema isolation (Phase 8, task 8.1)

Status: Accepted (2026-09-16)

## Context

Task 8.1 asks for Alembic migrations covering every table in CLAUDE.md §3,
with `make migrate` running from an empty DB and `make seed` loading
fixtures. CLAUDE.md §0.3 additionally requires tests to run against a
dedicated `comiccanvas_test` schema so they never touch dev data — already
the convention `PostgresMemoryStore`'s own tests (task 7.3, ADR 0014) use.
Making Alembic itself honor that convention surfaced three real,
independent bugs, only found by running migrations against a real
schema-scoped connection, not by reading Alembic's docs.

## Decision — raw SQL migrations, no ORM

Nothing in this codebase uses SQLAlchemy models (raw `psycopg` everywhere,
e.g. `app/db.py`); the one migration file
(`migrations/versions/af5692fa786e_initial_schema.py`) uses `op.execute(sql)`
exclusively, never `op.create_table`. `target_metadata = None` in `env.py` —
there is nothing to autogenerate against, and there won't be. SQLAlchemy
itself stays as an unavoidable transitive dependency of Alembic's engine
plumbing, not because this project uses its ORM.

## Bug 1 — `%` in a URL-encoded DSN breaks `configparser` interpolation

`Config.set_main_option()` writes through a `configparser.ConfigParser`
with `BasicInterpolation` enabled, which treats a bare `%` as the start of
a `%(...)` reference. A `DATABASE_URL` carrying a URL-encoded query string
(`?options=-csearch_path%3Dfoo`, needed for schema isolation, see Bug 3)
broke with `ValueError: invalid interpolation syntax` the first time one
was tried. Fixed by escaping every literal `%` to `%%` before calling
`set_main_option` — a one-line, easy-to-miss requirement `configparser`'s
own docs don't surface for this use case.

## Bug 2 — a schema-only `search_path` hides the `vector` type

Setting `search_path` to *only* the test schema (no `public`) made
`identity_models`'s `embedding vector(512)` column definition fail with
`psycopg.errors.UndefinedObject: type "vector" does not exist`, even
though the `vector` extension was already installed in `public`. This is
the identical bug class already fixed once in `app/db.py`'s `connect()`
(task 7.3) — `pgvector`'s type isn't visible unless `public` stays in the
search path. Fixed the same way: the test schema's `search_path` is always
`"{schema},public"`, never the schema alone.

## Bug 3 — Alembic's own version table silently resolves to `public`

The real one, and the reason this ADR exists. After fixing Bugs 1 and 2,
`alembic upgrade head` against a schema-scoped DSN reported success with
zero errors — but created **zero tables** in the target schema
(`information_schema.tables` returned nothing, even though the schema
itself genuinely existed). Root cause, confirmed by direct investigation
rather than assumed: Postgres resolves an *unqualified* table reference by
checking each schema in `search_path` order and using the **first one
where that table already exists** — not the first schema in the list
unconditionally. `public.alembic_version` already existed (with the
target revision already recorded, from an earlier non-isolated run), so
Alembic's own "is this migration already applied" check found and used
`public`'s copy instead of creating a fresh one in the test schema, and
silently skipped all migration work as a result.

Two other hypotheses were tested and ruled out first, live, before landing
on this one:

- **SQLAlchemy/psycopg not forwarding the `options` query param** —
  disproven by querying `SHOW search_path` through a raw engine built from
  the same URL; it correctly returned `"{schema},public"`.
- **The `%%`-escaping fix (Bug 1) corrupting the round-tripped URL** —
  disproven by directly inspecting both `Config.get_main_option()` and
  `Config.get_section()` (the actual path `env.py`'s `engine_from_config`
  call uses); both returned the un-mangled original URL.

Fixed with Alembic's own purpose-built `version_table_schema` parameter
(`context.configure(version_table_schema=...)`, confirmed wired through
`alembic/runtime/migration.py`), which schema-qualifies Alembic's version
table explicitly, sidestepping `search_path` resolution for it entirely.
`env.py` also has to `CREATE SCHEMA IF NOT EXISTS` the target schema itself
before this runs (with its own `commit()` — it isn't part of the
migration's own transaction), since nothing else creates it.

## Design — `DB_SCHEMA`, one env var, not per-test DSN surgery

`env.py` reads a `DB_SCHEMA` env var (unset in real dev/prod — those always
use the default `public` search path) and, when set, both appends
`?options=-csearch_path%3D{schema},public` to `DATABASE_URL` and passes
`version_table_schema={schema}` to `context.configure`. This keeps schema
selection in exactly one place rather than every test hand-building a
schema-qualified DSN itself.

## Verification

Real, live-verified end to end, not just unit-tested: `DB_SCHEMA=comiccanvas_test alembic upgrade head`
now genuinely creates all 16 real tables plus a schema-local
`alembic_version` inside `comiccanvas_test` (confirmed via
`information_schema.tables`), leaves `public` — including task 7.3's
`memory_beats`/`memory_people`/`memory_places` — completely untouched, and
`alembic downgrade base` cleanly removes exactly those 16. A manual `psql`
transaction with `SAVEPOINT` verified the real FK chain
(`users→jobs→days→beats→panels→candidates`) both accepts a valid insert
and rejects an invalid `job_id` with `ForeignKeyViolation`, then rolled
back with zero residue. `tests/integration/test_migrations.py` (3 tests,
`enable_socket`, skips without `DATABASE_URL`) automates all three checks
using the same `DB_SCHEMA` mechanism, isolated under its own
`comiccanvas_test_migrations` schema so it never collides with
`PostgresMemoryStore`'s own `comiccanvas_test` fixture. `make migrate`
(`alembic upgrade head`) and `make seed` (`scripts/seed.py` — 1 user, 7
days each with a beat/panel/candidate, cast of 3) both run for real
against the dev DB and are idempotent on re-run. `ruff`/`mypy --strict`
clean; full suite (468 tests) green via `make test`.

## Consequences

- Any future test needing an isolated schema (not just this migration)
  should reuse the same `DB_SCHEMA` convention rather than re-deriving its
  own `options=-csearch_path=...` DSN — that's exactly the fragile,
  bug-prone path this ADR documents fixing once.
- `memory_beats`/`memory_people`/`memory_places` (task 7.3) remain a
  separate, deliberately un-consolidated set of tables from the new
  `beats`/`panels`/`candidates` (see the migration file's own docstring
  for the full rationale) — this ADR doesn't change that.
- Task 8.1 is otherwise complete: `make migrate` from empty, `make seed`
  loading real fixtures. Remaining Phase 8 tasks (8.2 onward) build on top
  of this real schema.
