# 0016. Weekly recap graph (Phase 7, task 7.7)

Status: Accepted (2026-09-16)

## Context

Task 7.7 asks for a separate graph that takes 7 `BeatSheet`s, ranks all
beats by importance, builds a 6-panel `Script`, and reuses already-
chosen daily panels where a `beat_id` matches — generating only the
missing ones. Building this surfaced a contract limit, a real
frozen-design layout requirement, and a genuine data-loss bug in task
7.3's memory store that nothing had exercised until this needed to look
at beats across more than one day.

## Decision — widen `Script.panels` to 6, don't introduce a parallel type

CLAUDE.md's own task 7.7 wording ("produces a 6-panel `Script`") points
at reusing the exact daily `Script` contract, not a parallel weekly
shape. `Script.panels`'s `max_length` was 4 (task 3.1's daily 2-4 panel
rule); widened to 6. Daily scripts still only ever produce 2-4 panels in
practice — this just stops the contract itself from rejecting the
weekly case. Snapshot regenerated (`UPDATE_SNAPSHOTS=1`).

## Decision — 6-panel grid is 2x3, per the real frozen design

CLAUDE.md §0.1: reproduce the frozen design, don't invent new UI.
`design/comiccanvas-dashboard.html`'s own CSS has
`.strip.six{grid-template-columns:repeat(3,1fr)}` — 3 columns, i.e. 2
rows x 3 columns, used for exactly this recap. `compose.py`'s
`_grid_shape` (task 7.1) gained an explicit `6: (2, 3)` entry rather than
falling through to the generic `(1, n_panels)` default every other
unnamed count gets, which would have produced an unintended 1x6 strip.

## Decision — `DayState` as the carrier, `compose_strip` not `compose_day`

The per-panel generation helpers (`write_single_prompt`,
`generate_one_panel`, task 7.4/7.6) only ever read
`state.script`/`state.job_id`/`state.user_id` — building one synthetic
`DayState` (job_id derived from the ISO week, `beats` holding a
synthesized cross-day `BeatSheet`) lets the weekly graph reuse that
already-tested code completely unchanged, rather than generalizing
their signatures away from `DayState` for one caller. Composition uses
`compose_strip` (the pure function) directly instead of `compose_day` —
`compose_day` is hardcoded to the daily `strips/{user}/{date}/...`
prefix and resolves candidates from a whole `DayState`'s `chosen`
candidates, neither of which fits a recap's mixed reused/generated
candidate list or its own `weekly/{user}/{iso_week}/...` prefix.
Reusing `choose_best_for_panel` (task 6.7) per missing panel, not
`choose_best_candidates`, follows from the same reasoning: the weekly
graph decides reuse-vs-generate per panel itself, so it only needs the
single-panel selection/retry logic, not the whole-day orchestration
around it.

## Real bug found and fixed — `Beat.id` is only unique per day

The first thing that ever needed to look at beats across more than one
day surfaced this: `Beat.id` is only unique WITHIN one day's
`BeatSheet` — the extractor restarts numbering ("b1", "b2", ...) every
day (visible in every fixture this whole project has ever used).
`memory_beats`'s original uniqueness constraint (task 7.3) was
`UNIQUE (user_id, beat_id)` alone. Reproduced live before fixing:
writing a real Monday beat with id "b1", then a real Tuesday beat also
with id "b1", silently overwrote Monday's row — Monday's event became
unrecoverable under its own search terms. Fixed to
`UNIQUE (user_id, date, beat_id)`, with a real, idempotent migration in
`ensure_schema` (checks `pg_constraint` scoped to the *current* table
via `conrelid = 'memory_beats'::regclass`, not just by constraint name —
a first version of this migration wrongly matched a same-named
constraint belonging to a different schema's table, caught by running
it for real against the actual dev database, not just a fresh test
schema) that drops the old constraint and adds the new one, preserving
existing rows. `PostgresMemoryStore.write_beats`'s `ON CONFLICT` clause
and `InMemoryMemoryStore`'s internal `_panel_urls` keying were updated
to match. New `MemoryStore.get_panel_urls(user_id, beat_id, day)` — the
real read path task 7.7 needed and nothing had built yet — is
correctly day-scoped from the start.

## Verification

Real: `tests/unit/llm/test_weekly_graph.py`, 5 tests with every LLM
role, the image provider, and the four critic signals mocked (same
convention `test_graph.py` established) — covers the literal accept
line (a 7-day fixture with 4 of the top-6-importance beats already
having a real daily panel URL, exactly 2 without one, confirming
`new_generations == 2`), top-6-by-importance selection correctly
dropping the lowest-importance beat, direct URL reuse for the 4 already-
chosen panels, a valid 6-panel composed strip, and a clean error on an
empty week. Also rendered a real recap from real Gate 5 photos (4 real
reused panels + 2 real freshly-mocked-generated ones) and inspected it
directly — the 2x3 grid matches the frozen design exactly. The beat-id
collision bug was reproduced live against the real database both before
and after the fix (confirming the break, then confirming the repair),
and 4 new regression tests (2 in `test_memory.py`, 2 in
`test_memory_postgres.py`) guard against it permanently. `ruff`/
`mypy --strict` clean; full suite (466 tests) green via both direct
pytest and `scripts/test-all.sh`.

## Consequences

- Any future code that writes to or reads from `memory_beats` by
  `beat_id` alone (rather than `(date, beat_id)` or through
  `get_panel_urls`/`search`, both already fixed) will reintroduce the
  same class of bug — `beat_id` must never be treated as globally
  unique per user.
- Gate 7 can now close: tasks 7.1-7.7 are all real, tested, and
  documented. `services/worker` coverage and the combined daily+weekly
  integration run are the gate's own remaining literal check.
