# 0017. Replace Supabase (DB + Auth) with Neon + Clerk

Status: Accepted (2026-09-16)

## Context

CLAUDE.md §1 already decided "Postgres + pgvector (Supabase project in
dev; ... in prod)" and "Auth: Supabase email + password". ADR 0009
already substituted the *dev* Postgres with a native local install, to
defer the Supabase signup. Now, ahead of Phase 8's real auth (task 8.2)
and eventual Phase 12 deployment, the user hit Supabase's free-tier
project limit directly (most likely the "2 active projects per org, and
free projects pause after 7 days idle" caps — see below) and asked for
a free alternative for deployment, not just dev.

## Decision

Real, live-verified (each service's own current pricing page, 2026-09-16
— not training-data memory, which is stale on this kind of thing):

**Database — Neon**, replacing Supabase's Postgres role (both dev and
prod, superseding ADR 0009's dev-only local-Postgres stopgap once this
is wired up):
- Free tier: 0.5 GB storage, 100 compute-hours/month, autoscale to 8GB
  RAM, 10 branches/project, no credit card required.
- Native `pgvector` support (same extension task 7.3's `memory_beats`
  already depends on) on every plan, free included.
- The only real catch: compute scales to zero after 5 minutes idle —
  wakes automatically on the next query, no data loss, no deletion.
  Materially better than Supabase's own free-tier catch (projects
  **pause** after 7 days idle and need a manual dashboard resume, capped
  at 2 active projects per org — almost certainly what was actually hit).

**Auth — Clerk**, replacing Supabase Auth (task 8.2's JWT verification,
task 10.3's sign-in/sign-up pages, task 12.1/12.5's account setup):
- Free tier: 50,000 Monthly Retained Users (returns within 24h of
  signup), no credit card required — far beyond this project's scale.
- Real env vars (verified against Clerk's own Next.js quickstart):
  `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` (frontend), `CLERK_SECRET_KEY`
  (backend) — replacing `NEXT_PUBLIC_SUPABASE_ANON_KEY`/service-role
  patterns.
- JWT verification keeps the same *shape* task 8.2 already designed
  around (a cached JWKS endpoint) — Clerk exposes JWKS at the
  instance's Frontend API URL + `/.well-known/jwks.json`, the same
  pattern Supabase used (`SUPABASE_URL/auth/v1/.well-known/jwks.json`).
  `AUTH_MODE=mock` (already built, task 8.2) is unaffected — mock stays
  the default for all local dev through Phase 11, this only matters
  once real auth is actually wired up.

Options considered and rejected, all confirmed live before rejecting:
- **Railway Postgres**: no real free tier anymore — a one-time $5
  trial credit only, not a persistent free database.
- **Render Postgres**: free tier auto-deletes the database after 30
  days (14-day grace period, then gone) — unacceptable for a project
  meant to hold someone's real diary data.
- **Fly.io Postgres**: no standing free tier, just a 7-day/2-hour
  trial, then requires a card.
- **Firebase Auth / Auth0**: both have comparable or larger free tiers
  (50,000 / 25,000 MAU) and would have worked equally well; Clerk was
  picked for the cleaner FastAPI+Next.js pairing and JWKS setup that
  most closely mirrors task 8.2's already-designed Supabase JWT flow,
  minimizing the actual code change.

## Consequences

- CLAUDE.md §1's decisions table, §3, §6 env vars, and every task
  mentioning Supabase (0.2 historical note aside — that one stays as
  the real record of what was true then, per ADR 0009) are updated to
  name Clerk/Neon instead.
- Task 12.2's literal text named "ADR `0008-prod-database.md`" for the
  prod-database decision — that number was already taken (task 5.7's
  real style-card ADR) before task 12.2's text was ever updated to
  match. This ADR (0017) is the actual prod-database decision; task
  12.2 is corrected to reference it by its real number.
- ADR 0009 (dev-database-local-postgres) is not superseded for *local
  dev* — local Postgres stays the dev default (matching CLAUDE.md's own
  "no deploys mid-build" rule); Neon only comes into play at Phase 12,
  or sooner if a shared/persistent dev database is ever wanted.
- No code changes needed right now — `AUTH_MODE=mock` and local
  Postgres already carry all of Phase 8-11's development. This is a
  documentation-only decision until Phase 12 (or task 8.2/10.3
  specifically) actually implements it.
