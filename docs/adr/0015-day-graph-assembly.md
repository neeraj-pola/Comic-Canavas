# 0015. DayGraph: pipeline assembly with LangGraph (Phase 7, task 7.4)

Status: Accepted (2026-09-16)

## Context

Task 7.4 wires nodes 3-10 into one real, connected pipeline. Every node
module until now documented itself as "exercised directly, `graph.py`
doesn't exist yet" (see `nodes/generate.py`/`nodes/critic.py`'s own
docstrings). LangGraph (CLAUDE.md §1's already-decided orchestration
layer) had zero prior usage in this codebase — its real current API was
verified live (small running scripts, not just docs, since the docs
site's rendered pages didn't fetch cleanly) before any design decision
relied on it.

## Decision — no Send-based fan-out; wrap the existing async functions

`generate_panels` (task 4.5) already fans out across all panels via
`asyncio.gather`; `choose_best_candidates` (task 6.7) already loops per
panel with its own retry-then-fallback, calling an injected `regenerate`
callback. LangGraph's idiomatic map-reduce pattern (`Send`, verified
live) would fan out at the *graph* level instead — but that would mean
rebuilding retry/fallback logic that's already built, tested, and
correct. `generate` and `critic` are each one graph node wrapping the
existing function unchanged. The panel-scoped retry is a plain callback
composed from two new functions added specifically to support it:
`nodes/prompts.write_single_prompt` (rewrites one panel's prompt) and
`nodes/generate.generate_one_panel` (generates candidates for one
already-built prompt) — both deliberately duplicate a few lines of setup
from their whole-day counterparts (`write_prompts`/`generate_panels`)
rather than refactoring those into a shared path, keeping zero risk to
their existing tests.

## Real API finding — reducers double-count full-state returns

Every existing node function reads a whole `DayState`, computes complete
new values (including any accumulation the node needs — e.g.
`choose_best_candidates` already does `all_errors = list(state.errors);
...; all_errors.extend(new)` itself), and returns a full `DayState` via
`model_copy(update=...)`. Verified live, not assumed: giving a field like
`errors` an `Annotated[list[str], operator.add]` reducer causes LangGraph
to treat the ENTIRE returned list as a fresh contribution on every node
hop — a node that returns `errors` completely unchanged still gets it
appended again, silently doubling accumulated fields. Fix: no field on
`DayState` uses an accumulating reducer; every field uses LangGraph's
default replace-on-set behavior, matching how every node already
computes and returns complete values. `contracts.DayState` needed zero
changes to work as `StateGraph`'s state schema.

## Real gap closed — nothing was scoring candidates

`nodes/critic.py`'s own docstring states scores are "already computed
... by whatever called this node." Nothing did. Tasks 6.1-6.4 built and
unit-tested `critic/identity.py`/`style.py`/`alignment.py`/`detail.py`
in isolation; task 6.8 built the OCR guard the same way. This is the
first place they run together against a real stored candidate image.
New `nodes/score.py`: `score_one_candidate` fetches real bytes via
`Storage.get_object_by_url` (task 7.1) and runs all four signals plus
the text-detection guard; `score_candidates` does this for a whole
`DayState`, skipping any candidate that already has scores (so a
resumed run, task 7.6, doesn't re-pay real cost). `_critic` calls it
before `choose_best_candidates`; the `regenerate` callback scores each
freshly generated candidate the same way, so a retried candidate is
judged identically to the first batch. Kept as its own module rather
than folded into `critic.py`, so `choose_best_candidates` — already
correct — never had to change.

## Scope boundaries (matching CLAUDE.md's own task list, not improvised)

- Node 2 (ASR, task 8.3) isn't built. `_input` copies `text` into
  `transcript` for `source="text"` days; `source="audio"` raises a
  clear `AsrNotImplementedError` rather than mishandling it silently.
- Node 9 (feedback) is API-triggered post-hoc (task 8.6) — not a step
  in this graph. Nodes 11/12 (weekly training, eval gate) are Phase 9.
  The weekly recap graph (task 7.7) is separate.
- Quiet-day/sensitive-day handling (task 7.5) needed no new graph
  structure: `write_script` already produces 2-panel quiet scripts, and
  every downstream node already generalizes to panel count 2-4
  (`compose.py`'s `_grid_shape`, task 7.1). Task 7.5 tests that path
  through this same graph; it doesn't build a different one.
- Per-node wall-clock timing is recorded into `DayState.versions`
  (`"{node}_ms"`) as a lightweight stand-in for CLAUDE.md's literal
  "timing to `jobs.events` (JSONB)" — the real `jobs` table is task 8.1,
  which doesn't exist yet. Deferred, not silently dropped.

## Verification

Real: `tests/unit/llm/test_graph.py`, 5 tests running the actual
compiled graph end-to-end (`input -> beats -> script -> prompts ->
generate -> critic -> compose -> memory`) with every LLM role and the
image provider mocked, and the four critic signals mocked at
`nodes/score.py`'s call boundary (patched where imported to, not where
defined — they're unit-tested against real models elsewhere in Phase 6;
this test is about orchestration). Covers: a full run produces a
4-panel `layout.json` with exactly one chosen candidate per panel
clearing the real identity threshold; completes in ~1.1s per run (well
under the 3s accept line); per-node timing lands in `versions` for every
node; `source="audio"` raises the documented not-implemented error
instead of misbehaving; beats get written to memory with real per-panel
URLs attached. One real test bug caught and fixed along the way: the
mock LLM extractor always returns its own default canned `BeatSheet`
(about a dinner/sunset beat) regardless of `DayState.text` — an
end-to-end test that searches memory for a word from the *input* text
rather than the *actual mocked output* fails for a reason that has
nothing to do with the graph. `ruff`/`mypy --strict` clean; full suite
(452 tests) green via both direct pytest and `scripts/test-all.sh`.

## Consequences

- `nodes/prompts.py` and `nodes/generate.py` each gained one new public
  function (`write_single_prompt`, `generate_one_panel`); their existing
  whole-day functions (`write_prompts`, `generate_panels`) are untouched.
- Any future node added to the daily pipeline should follow the same
  shape already established: take a whole `DayState`, return a complete
  new one, do its own internal accumulation — never rely on a LangGraph
  reducer to accumulate across hops.
- `DayGraph` requires a `LookCard` up front (not optional) — scoring
  genuinely cannot run without one, unlike `write_prompts`, which
  degrades gracefully to a less-personalized prompt when it's missing.
- Task 7.6's idempotency (resuming from persisted `DayState`) has a real
  head start here: `score_candidates` already skips already-scored
  candidates, and `write_beats`'s upsert (task 7.3) already tolerates
  being called twice with the same `beat_id`.

## Addendum — 2026-09-16, task 7.5 (quiet-day / sensitive-day paths)

Confirmed rather than assumed: `compose.py`'s `_grid_shape` (task 7.1)
and the critic's per-panel loop (task 6.7) already generalize correctly
to panel count 2, not just the 4 every other test in this file exercises
— both new tests (`test_graph_day_types.py`) passed on the first run
with no code changes needed. Real scope boundary, stated in that file's
own docstring: with every LLM role mocked, "humor is suppressed for a
sensitive day" isn't something a graph-level test can verify — that's
real model behavior, already covered by `ml/evals/script.py`'s Gate 3
judge eval and task 2.5's sensitive-flagging accuracy. What's real here
is that a genuinely flagged `BeatSheet` (`too_short`/`sensitive`) flows
through `extract_beats` unchanged and a 2-panel script reaches a valid
composed strip with no architecture assuming 4 panels anywhere.

## Addendum — 2026-09-16, task 7.6 (idempotency)

Real Postgres-backed checkpointing via `langgraph-checkpoint-postgres`'s
`AsyncPostgresSaver`, against the same local Postgres task 7.3 already
uses (ADR 0009). Two more real findings, verified live with throwaway
scripts before any design decision relied on them (same discipline as
0015's own `Send`/reducer findings above):

1. **Resuming needs `None`, not the caller's natural input.** Calling
   `.ainvoke(state, config)` a second time with a fresh, non-`None`
   `state` — the obvious thing a caller does after a restart, having no
   reason to know a special convention exists — silently RE-RUNS every
   node from `START`, including ones a checkpoint already completed
   (verified: a node's call counter incremented again on the second
   call). `DayGraph.run` calls `await self._compiled.aget_state(config)`
   first — `bool(snapshot.values)` is `False` for a `job_id` with no
   prior checkpoint and `True` once one exists — and only passes `None`
   into `ainvoke` in the resume case. Without this check, a checkpointer
   could be fully configured and correctly persisting state, and the
   accept line ("no second extractor call") would still silently fail.
2. **`AsyncPostgresSaver.setup()` does not create its own schema.**
   Pointing a schema-scoped DSN (`?options=-csearch_path=...`, itself
   verified live as a working way to isolate checkpoint tables the same
   way task 7.3 isolates `memory_*` tables) at a schema that doesn't
   exist yet fails with `psycopg.errors.InvalidSchemaName: no schema has
   been selected to create in`. `open_day_graph` ensures the schema
   first, directly against the same `resolved_dsn` it actually uses —
   not routed through `app.db.connect`, which always resolves its own
   DSN from `Settings` regardless of `open_day_graph`'s own `dsn`
   override and would silently create the schema in the wrong database
   if the two ever diverged. `app.db`'s private schema-name validation
   was promoted to a public `validate_schema_identifier` so both places
   that ever interpolate a schema name into SQL/a connection string
   share one rule instead of two copies drifting.

`open_day_graph` is an async context manager (mirroring
`AsyncPostgresSaver.from_conn_string`'s own shape — its connection must
stay open for as long as the graph is used) that opens the checkpointer,
ensures the schema, calls `.setup()` (idempotent, safe on every call),
and yields a ready `DayGraph`. `DayGraph` itself is unchanged for
callers that don't pass a `checkpointer` (task 7.4's original default) —
task 7.6 is purely additive.

**Real verification**: `tests/unit/llm/test_graph_idempotency.py`, 3
tests against the real local Postgres in a dedicated `comiccanvas_test`
schema (`enable_socket`, skip-if-no-`DATABASE_URL`, matching
`test_memory_postgres.py`'s own pattern). Covers the literal accept
line — kill the graph mid-`script` (a real raised exception, not
simulated by mocking around the checkpoint mechanism), rerun the same
`job_id`, and confirm the extractor's call count stayed at 1 while
script's incremented to 2 (it's the one that failed and correctly
reruns) — plus a fresh `job_id` running normally with a checkpointer
configured, and two different `job_id`s not interfering with each
other's resume state. `ruff`/`mypy --strict` clean; full suite (457
tests) green.
