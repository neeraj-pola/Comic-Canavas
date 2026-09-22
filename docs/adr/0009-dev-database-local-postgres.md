# 0009. Dev database: native local Postgres instead of Supabase, for now

Status: Accepted

## Context

CLAUDE.md §1 already decided "Postgres + pgvector (Supabase project in
dev; ...)" — reopening it needs an ADR (§0.1). The user asked for a
substitute for the dev Supabase project specifically, to defer that
signup for now. This machine already runs a Homebrew-installed Postgres
15 (for an unrelated project); pgvector's Homebrew bottle only targets
Postgres 17/18, so it doesn't install against 15 out of the box.

## Decision

Use that existing local Postgres 15 for dev instead of Supabase, with
pgvector built from source against it (`PG_CONFIG=.../postgresql@15/bin/pg_config
make install`, from the pgvector v0.8.0 source — a normal, documented path
for a Postgres version newer bottles don't cover) rather than installing
a second Postgres version. A dedicated `comiccanvas` role/database keeps
it isolated from the other project's database on the same server —
`DATABASE_URL=postgresql://comiccanvas@localhost:5432/comiccanvas`, local
trust auth, no password. No code changes: `DATABASE_URL` is just a
connection string: `LocalFileStorage`'s storage backend, `Storage`
interface, and now the DB are the same "interface first, real backend
swappable" pattern.

## Consequences

`AUTH_MODE=mock` stays as-is (already independent of the DB choice) — no
Supabase Auth either, for now. This doesn't touch prod: ADR 0008 (task
12.2) still decides between Railway Postgres and a real Supabase DB for
prod, unaffected by this. Task 0.2 is satisfied by this substitution (its
Accept line — `make dev` starts all three, `/health` reports db + redis
reachable — is true); its literal text ("create a Supabase project")
isn't followed, by request. Whoever picks this up on a fresh clone needs
to know this convention isn't in a fresh Homebrew Postgres install by
default — see the updated README setup section.
