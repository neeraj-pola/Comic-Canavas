# 0007. Preference optimization — part 1: reward model, registry, and eval gate (Phase 9, tasks 9.1-9.2, 9.6-9.10)

Status: Accepted (2026-09-16, addendum 2026-09-16) — partial: this is the
reward-model/gate/registry/prompt-persistence half of Gate 9's own
deliverable name (§5, "docs/adr/0007-preference-optimization.md"). Script
DPO (9.3) and diffusion-DPO (9.5, and stale as literally written — see
below) are real, separately-scoped work this ADR's own "Explicitly not
attempted this session" section covers; task 9.4 is reframed below to an
offline form whose plumbing (durable prompt history) is now real but
whose training code still has no real data to run on. This file will be
extended (not replaced) once those land and Gate 9 fully closes.

## Context

Phase 9's whole premise — learning from real preference data — has a
real, structural dependency this session had to confront directly: every
training step needs real A/B picks, caption edits, and thumbs from real
daily use, and none existed yet (`image_pairs`/`caption_pairs`/`thumbs`
were empty; Phase 10's frontend, which people would actually use, isn't
built). Rather than block Phase 9 entirely on Phase 10 + 7 days of real
personal use (CLAUDE.md's own Gate 10 dependency), the user asked for a
synthetic week of data instead: 7 real days (2026-09-14 through
2026-09-20) run through the real pipeline (`app.worker.run_daily_job`,
real Claude/GPT-4o-mini calls, `IMAGE_PROVIDER=mock` for free image
generation) with real 4-panel scripts, plus synthesized feedback
(28 real `image_pairs`, 28 `thumbs`, 5 `caption_pairs`) derived from the
real critic scores those mock candidates actually got.

## Decision — that synthetic week is CLAUDE.md §4's held-out week, frozen for real

`ml/goldens/pairs/heldout_week.jsonl` (§4: "first real week of A/B taps")
didn't exist. Rather than invent a second, parallel concept, this
session froze the real 28 pairs (with each candidate's real
identity/style/alignment/detail scores) into that exact file. This
resolves a real ordering question cleanly: `export_feedback` (task 9.1)
now excludes any row whose day falls in an ISO week present in this
file — by the row's own date, not by which week the cron happens to run
in (a real bug caught before shipping: a boundary-only check would leak
held-out rows into a later export once a second week exists).

## Decision — the reward model (task 9.2) needs no GPU

CLAUDE.md names "Modal A10G" for this task. A Bradley-Terry head over
four already-computed scalar features (identity/style/alignment/detail —
task 6.1-6.4's real outputs, already in `candidates.scores`) is exactly
`nn.Linear(4, 1, bias=False)` — logistic regression on a 4-dimensional
feature difference. Training it on 28 pairs takes milliseconds on CPU;
provisioning a GPU would be pure overhead. `ml/reward_model/train.py`
trains it for real, exports real ONNX (`torch.onnx.export(...,
dynamo=False)` — torch 2.14's new default exporter needs `onnxscript`,
not installed; the legacy TorchScript exporter needs nothing extra and
is more than adequate for one linear layer, a real bug found live, not
a style choice), and reports pairwise accuracy against the hand-set
head (`services/worker/app/critic/reward_head.py`, task 6.5).

**Real, honest result**: `learned_pairwise_accuracy` and
`hand_set_pairwise_accuracy` both landed at 0.4286 on the held-out
split — no improvement. This is a genuine, expected consequence of the
underlying data, not a bug: `IMAGE_PROVIDER=mock` candidates are flat
placeholder rectangles, so `identity` is a constant 0.15 for every
candidate (nothing for a difference-based model to learn from) and
`detail` has almost no real variance either; only `style`/`alignment`
differ at all, and only slightly. The eval gate (task 9.7) correctly
rejects this result — recorded for real on `training_runs.decision`
(`reject`, reason: "reward accuracy 0.4286 does not beat champion 0.4286
by >= 0.01"). The mechanism works exactly as designed; a second, real
week of actually-varied candidates (real generation, real human taps) is
what would give it something real to learn.

## Decision — registry reuses task 5.6's real HF push, not a new one

`ml/registry.py`'s `register_checkpoint`/`list_checkpoints` wrap
`ml/identity/hf_registry.py`'s `upload_lora` (task 5.6, already real and
live-verified against the project's actual private HF repo) rather than
reimplementing an HF client — uploading arbitrary bytes to a path in
that repo was never actually LoRA-specific. The reward model's real
ONNX checkpoint was registered for real: pushed to
`https://huggingface.co/Neeraj123456/comiccanvas-models/resolve/main/reward_model/reward_head_v1.onnx`
and confirmed present via `HfApi.list_repo_files` — not just asserted
from a mocked test.

**Real bug found and fixed**: `register_checkpoint` called
`upload_artifact` (which reads `HF_REPO`/`HF_TOKEN` from `os.environ`
directly, no `.env` loading of its own) before anything had populated
the real process environment, raising `MissingHfConfigError` even with
a real `.env` on disk. Fixed by calling `load_dotenv` first, at the top
of `register_checkpoint`.

**Real test-hygiene bug found and fixed**: this session's first version
of `kick_training` (task 9.9's own scheduler entrypoint) called the real
`ml.reward_model.train.main()` — and so did its first unit test,
un-mocked. Every test run silently retrained the reward model against
the real dev DB and re-pushed a file to the real HF repo, exactly what
CLAUDE.md §0.3's "tests use a dedicated schema so they never touch dev
data" rules out. Caught by noticing extra rows in `checkpoints` after a
routine test run, not by design. Fixed two ways: the test now mocks
`ml.reward_model.train.main`, and — defense in depth — `ml/reward_model/
train.py` and `ml/registry.py`'s own DB connections now respect a
`DB_SCHEMA` env var the same way `services/api/app/db.py` and `services/
worker/app/worker.py`'s `_connect()` already do, so a future test that
forgets to mock the call still can't reach `public` by accident. The two
real rows and stray local report file this created were deleted by hand;
no extra file was pushed to HF (only `register_checkpoint`, called
directly, pushes to HF — `main()` on its own does not).

## Decision — eval gate and feature flags are pure, GPU-free logic

`ml/evals/gate.py`'s `decide()` implements CLAUDE.md's literal four
rules (reward accuracy margin, script win-rate floor, critic-mean
non-regression, cost-increase ceiling) as one pure function over a
`GateInputs` dataclass — no model-specific knowledge, unit-tested
against both synthetic edge cases (task 9.7's own "a synthetic
regression is rejected" accept line) and this session's real reward-
model result. `record_decision` writes the real decision + reasons onto
`training_runs`.

Task 9.9's flags (`WRITER`/`PROMPTER`/`REWARD_HEAD`/`GENERATOR`, CLAUDE.md
§6) are read once per job in `graph.py`'s `_input` node and recorded into
`DayState.versions` — real, but honest about what's behind them: only
`base`/`hand` are real paths today (`dpo`/`grpo` have nowhere to route
to until tasks 9.3/9.4 exist); `GENERATOR` is recorded as an alias of
the already-real `IMAGE_PROVIDER` switch (task 4.1), not a second,
independent mechanism that could drift from it.

## Decision — live rollback detects and alerts for real; the flip is a real, named gap

`services/scheduler/app/cron.py`'s `first_pick_rate_by_day`/
`is_rollback_needed`/`live_rollback` compute a real 3-day first-pick
rate from real `candidates` data and alert (via the same
`NotificationAdapter` `remind` uses) when it drops more than 5 points
below a baseline. "Flip the flag back to champion" has no real target
yet: `WRITER`/`PROMPTER`/`REWARD_HEAD` are read from `os.environ` by the
worker process at each job's start, and nothing lets a *different*
process (the scheduler) mutate that, nor does any DB-backed flag store
exist to make the flip safe across a restart. Detection and alerting are
real; the actual flip needs a small new `system_flags`-style table the
graph reads instead of/before `os.environ` — named here as a real
follow-up, not silently implied to already work.

## Decision — experiment tracking is real but unverified against a real account

`ml/experiment_tracking.py`'s `log_run` calls real `wandb.init/log/
finish` when `WANDB_API_KEY` is set; with no key (this project's current
real state — `.env`'s entry is commented out), it falls back to a real,
inspectable local JSON file under `ml/experiment_runs/` rather than
silently doing nothing. Matches this project's own precedent (task
4.4's `fal_flux.py`) for "code-complete, not live-verified, honestly
labeled" rather than either skipping it or claiming a false verification.

## Decision — task 9.5's literal premise is stale; task 9.4 is reframed around it

While scoping 9.3-9.5's GPU work (not run — see below), a real
inconsistency surfaced between CLAUDE.md's Phase 9 text and its own
Phase 5 decisions, caught by the user directly rather than by me:
task 9.5 as literally written ("Flux identity LoRA fine-tuned on image
pairs") assumes there is still a Flux LoRA in the production path to
fine-tune. There is not. ADR 0005 (task 5.15) already ran that exact
gate and rejected it: the character LoRA scored 0.00 mandatory-checklist
pass at every one of 9 checkpoints across two full training runs, and
`IdentityModel.generation_path = "reference"` won for real — production
image generation is `images/flux_kontext.py` calling `fal-ai/flux-2/edit`,
a **hosted, non-fine-tunable editing API**, conditioned on `master.png` +
a style reference image. Diffusion-DPO needs gradient access to the
diffusion model's own weights (it's a loss over the model's score-
matching objective, not over sampled text like DPO/GRPO); a hosted API
behind an HTTP call offers no such access, and the one local diffusion
model this project ever trained (the LoRA) is the thing ADR 0005 already
discarded. Task 9.5 as written has no real target left to apply
Diffusion-DPO to — it is stale, not merely unattempted.

The mechanism the user actually described when asking about this —
"map the prompt that produced the panel someone picked vs. the one they
rejected, and learn which kind of prompts get chosen" — is real,
implementable without GPU training, and is task **9.4** (prompt-writer
preference learning), not 9.5. CLAUDE.md's literal 9.4 text (live GRPO
rollouts: sample 8 prompts per panel, generate fresh images with Flux
Schnell, score with the critic, policy-gradient update a 3B model) is
the *online* form of this idea — it needs a trainable local policy model
and fresh paid generation calls during training. The *offline* form —
direct preference optimization (DPO's actual pairwise loss, not GRPO's
group-rollout one) over historical `(chosen_prompt, rejected_prompt)`
pairs recovered by joining a day's `image_pairs`/`thumbs` back to the
`ImagePrompt` that produced each side — needs no live generation and no
GPU, only the historical prompt text this pass just made durable (below).
Task 9.4 is reframed to that offline form; CLAUDE.md's task 9.4 wording
is updated accordingly (online GRPO stays named as a possible later
upgrade path once training infra exists for it).

## Decision — `image_prompts` persistence (plumbing for the reframed 9.4)

Before this pass, `ImagePrompt` (packages/contracts) was pure in-memory
`DayState` data — CLAUDE.md §3 never listed a table for it, so nothing
survived past one job's `DayState`, and `GET /days/{date}}`'s `prompts`
field was hardcoded `[]` since task 8.5 was written. That meant zero real
`(chosen_prompt, rejected_prompt)` history exists yet for the reframed
9.4 to train on, regardless of GPU/cost questions. This pass closes that
specific gap, and only that gap — no training code, no GPU spend:

- Migration `582963a9d783` adds `image_prompts` (id-keyed, not
  `(day_id, panel_id)`-keyed — see below) and `candidates.prompt_id`
  (FK, nullable).
- `graph.py`'s `_critic` node now accumulates every prompt actually used
  for a panel — including a retry's — into `DayState.prompts`, not just
  the original batch's. Task 6.7's retry design ("a retry discards the
  failed batch entirely") means only the *last* prompt per panel_id ever
  produced a surviving candidate; keeping both makes that visible instead
  of silently losing the discarded one.
- `persist.py`'s new `persist_image_prompts` keeps only the last prompt
  per `panel_id` (later entries — retries — win), writes it with a
  deterministic `{day_id}-p{panel_id}` id, and returns a
  `panel_id -> prompt_id` map that `persist_candidates` uses to link each
  surviving candidate to the prompt that actually produced it
  (`ON CONFLICT ... prompt_id = COALESCE(EXCLUDED.prompt_id,
  candidates.prompt_id)` — a later write can't accidentally null out an
  already-linked prompt). `worker.py`'s `regenerate_panel_job` (task 8.6)
  does the same for a live regenerate call.
- `GET /days/{date}` (`routers/days.py`) now reads `image_prompts` back
  for real instead of returning the hardcoded `[]`; `candidates` in the
  same response now includes `prompt_id`.

**Real, live-verified, not just unit-tested**: `tests/unit/llm/
test_worker_jobs.py`'s existing real end-to-end test (real Postgres, real
`alembic upgrade head`, `app.worker.run_daily_job` run directly, no
mocks below the LLM/image/scoring boundary) now also asserts every
persisted candidate has a non-null `prompt_id` and all 4 panels have a
row in `image_prompts` with real prompt text — this is the same test
that already proves the whole daily pipeline reaches `status='done'`,
extended rather than duplicated. `tests/unit/api/test_days.py` gained a
new test seeding a real `image_prompts` row + linked `candidates.
prompt_id` and asserting `GET /days/{date}` returns it. `tests/unit/llm/
test_graph.py` gained `test_day_graph_records_the_retry_prompt_not_the_
discarded_original`, forcing panel 1's original 3-candidate batch to fail
the identity gate and confirming both the discarded original prompt and
the surviving retry prompt land in `result.prompts` (2 entries for panel
1, not 1) — the mechanism `persist_image_prompts`'s "last one wins"
logic depends on. `tests/integration/test_migrations.py` extended with a
real FK-violation check (a candidate can't reference a non-existent
`prompt_id`). Full suite: 553 tests green (551 → +2 net after this pass's
additions), `ruff`/`mypy --strict` clean.

No offline-DPO training code was written this pass — there is still
exactly zero real `(chosen, rejected)` prompt-pair history (the 7-day
synthetic week and its 28 `image_pairs` predate this table's existence,
so no backfill is possible; the text was never durably stored anywhere
to recover it from). That training code is real, separately-scoped work
for whenever real usage — Phase 10, or another synthetic pass run after
this table existed — has produced pairs to actually learn from; writing
it against zero rows would be untestable in any way that means anything.

## Explicitly not attempted this session

Tasks 9.3 (script DPO, TRL + PEFT QLoRA on Qwen2.5-7B) and 9.5
(diffusion-DPO — and, per the decision above, stale as literally written
regardless) need real Modal GPU compute and non-trivial real cost — a
different category from anything else built this session. Task 9.4,
reframed above to its offline-DPO form, needs no GPU but does need real
historical prompt-pair data this pass doesn't yet have (see above) — its
training code is deferred for that reason, not cost. These are left as
real, explicitly-scoped backlog items pending the user's go-ahead (9.3)
or real data (9.4), not silently skipped or faked. Task 9.11 (optional
PPO ablation, "for the write-up") is deferred for the same reason
CLAUDE.md itself marks it optional.

## Verification

Real, not just unit-tested: the reward model was actually trained on the
actual frozen held-out file, actually exported to ONNX, actually
registered to the actual private HF repo (confirmed via a live
`list_repo_files` call), and actually gated (real `reject` decision
recorded on a real `training_runs` row) — end to end, once, for real,
before any test was written to cover the mechanism generically. 42 new
tests across `ml/reward_model/`, `ml/registry.py`, `ml/evals/gate.py`,
`ml/experiment_tracking.py`, `services/scheduler/app/cron.py`'s new
functions, and one new `graph.py` feature-flag test. `ruff`/`mypy
--strict` clean; full suite (551 tests) green via `make test`.

## Consequences

- `ml/goldens/pairs/heldout_week.jsonl` is now a real frozen golden file
  under CLAUDE.md §0.1's own rule — changing it needs a `golden:`-titled
  PR with a reason, same as every other file under `ml/goldens/`.
- The reward model's real "does not beat champion" result stands as the
  current champion is still the hand-set head (task 6.5) — no promotion
  has happened, correctly.
- Any future real week of usage data automatically becomes eligible
  training data (not held out) purely by not being the frozen week —
  `export_feedback` needs no further changes to pick it up.
- Live rollback's real gap (no flag-flip mechanism) means an actual
  production regression today would only alert, not self-heal — worth
  closing before this matters for real traffic.
- CLAUDE.md's task 9.5 wording itself is now understood to be stale
  (assumes a fine-tunable local diffusion model that Phase 5 already
  rejected) rather than merely unattempted — recorded here so a future
  reader doesn't try to execute it as literally written.
- Any future real or synthetic usage pass automatically produces durable
  `image_prompts` rows from this point forward — the offline-DPO training
  code for the reframed task 9.4 becomes runnable the moment enough
  `(chosen, rejected)` prompt pairs exist, with no further schema work.
