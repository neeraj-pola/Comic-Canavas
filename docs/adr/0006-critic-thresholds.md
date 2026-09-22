# 0006. Consistency critic thresholds (Phase 6)

Status: Accepted (Phase 6 complete 2026-09-15, tasks 6.1-6.8; Gate 6 passed)

## Context

CLAUDE.md's Phase 6 assembles four scoring signals (identity, style,
alignment, detail) into a Bradley–Terry reward head (task 6.5) used to
pick the best of several generated candidates per panel. Task 6.1
(identity) was written before ADR 0005's character-first pivot and
literally describes ArcFace cosine similarity to a reference photo, with
an accept line framed as separating same-person from different-person
selfies.

## Decision — 6.1 identity

ADR 0005 already demoted ArcFace to a weak secondary signal for Phase 5
(it can't reliably detect faces in stylized comic art, and the VLM
feature checklist, task 5.9, is both more answerable and more aligned
with what a comic diary's identity gate should measure). The critic's
identity score inherits that same decision rather than reopening it:
`critic/identity.py`'s `score_identity` is **checklist-primary (weight
0.7) + weak ArcFace agreement (weight 0.3)** — the same weighting
`ml/identity/master.py`/`sheet.py` already established for ranking
master/sheet candidates in Phase 5, reused here for ranking per-panel
generation candidates (task 4.5's 3-per-panel fan-out). A "no face
detected" result is not disqualifying on its own (same real bug class
already fixed in `master.py`'s `rank_candidates`) — the checklist still
runs on the full image.

The accept line's "AUC ≥ 0.9" is kept as the real metric, computed
against a same/different-**character** golden set instead of same/
different-*person* selfies, which don't exist for a designed comic
character and wouldn't mean anything for one. Positives are the approved
character's own real sheet/master crops (task 5.10/5.8); negatives are
the task 5.7 regularization set's different, generic comic characters —
both real, already-committed assets, so no new golden-data collection
was needed.

**Real numbers (2026-09-15, `ml/evals/critic_identity.py`,
`ml/evals/reports/critic_identity_auc_2026-09-15.json`)**: **AUC =
1.000** (target ≥ 0.90) — perfect separation. Mean positive score 0.738,
mean negative score 0.367, n=20 positives / 20 negatives.

## Decision — 6.2 style

CLAUDE.md's literal spec seeds the style reference set from `ml/
goldens/style/` (40 in-style + 20 off-style panels), which doesn't exist
yet — no chosen-panel history exists to curate it from, and this pass
didn't manufacture one. `critic/style.py` instead reuses the task 5.7
regularization bank (150 real in-style renders, already committed) as
both the scoring reference *and* the golden set's positive class,
directly via `ml/identity/style_score.py`'s existing top-k cosine scorer
— no new module logic needed beyond a thin, node-7-facing wrapper.

For the golden set's negative (off-style) class, the person's own raw
onboarding photos (real photographs, definitively off-style — already on
disk, task 5.1) serve as free, real negatives: a photograph is about as
far from "flat ink, bold outline, halftone" as an image can be, so this
is a genuine, not manufactured, off-style class.

**Real numbers (2026-09-15, `ml/evals/critic_style.py`,
`ml/evals/reports/critic_style_accuracy_2026-09-15.json`)**: **accuracy =
1.000** (target ≥ 0.90) at the best-sweep threshold 0.532. Mean positive
score 0.668, mean negative score 0.226, n=20 positives / 20 negatives.
Zero API cost — DINOv2 embedding runs locally on CPU.

## Decision — 6.3 alignment

Built as specified — SigLIP text-image similarity, no ADR-level
deviation from the mechanism itself. Two real things worth recording
because they cost real debugging time: `google/siglip-base-patch16-224`'s
API was verified live against HuggingFace's own docs before writing any
code (`padding="max_length"` is required or the score silently changes,
not errors; the real similarity is `sigmoid(logits_per_image)`, matching
SigLIP's pairwise training objective, not CLIP's softmax convention) —
and `SiglipTokenizer` needs `sentencepiece`, found only by actually
running it live, not documented anywhere obvious beforehand. Also fixed
while here: `services/worker/pyproject.toml` never declared `opencv-
python-headless`/`torch`/`transformers` even though task 5.14's
`refine_face.py` already imported `cv2` directly — it only worked via
the shared uv workspace venv. Declared honestly now rather than left as
an incidental dependency.

No dedicated alignment golden set exists (CLAUDE.md doesn't name one for
this task specifically). Used the 20 real Gate 5 panels (task 5.13 — real
images with known real captions) against a deterministic shuffle (each
panel's image paired with the *next* panel's caption, not a random one,
so the result is reproducible) rather than fabricate synthetic goldens.

**Real numbers (2026-09-15, `ml/evals/critic_alignment.py`,
`ml/evals/reports/critic_alignment_winrate_2026-09-15.json`)**: **win
rate = 0.950** (19/20, target ≥ 0.90). The one loss (panel 2, "pouring
coffee") had both the correct and shuffled captions score near-zero in
absolute terms — SigLIP was trained on real photography, and this
project's flat-ink comic style is a genuine domain shift, so absolute
confidence is low across the board. The *relative ranking* (correct >
wrong) still held 19/20 times, which is what this accept line measures,
not absolute confidence.

## Decision — 6.6 ONNX export and CPU benchmark

Investigated rather than assumed. Real findings, in order of how much
they actually matter:

1. **Identity's real signal is out of scope for a CPU benchmark at
   all.** Since this ADR made the VLM feature checklist the real
   identity gate (not ArcFace), identity scoring now requires a remote
   LLM API call per candidate — not a local model, can't be ONNX-
   exported, and its latency is network-bound (typically 1-3s per call),
   which would swamp any CPU-bound number if included. This benchmark's
   scope is therefore style + alignment + detail only.
2. **The actual dominant cost is already ONNX** and isn't one of
   CLAUDE.md's named four models: face detection (InsightFace/buffalo_l,
   needed for the detail critic's face-exclusion region) measured at
   ~2.1-3.1s of a ~5.6-7.1s total across 12 real candidates — already
   onnxruntime-based, nothing further to export. Tested downscaling the
   detector's input as a cheap speedup lever; measured **no real
   effect** (`det_size=640` internally resizes regardless of input size
   — 2.10s vs 2.15s, within noise) — a real negative result, not
   assumed away.
3. **SigLIP (alignment, ~1.7s) has no robust ONNX export path.** Checked
   `optimum`'s real, current support live rather than guessing: SigLIP
   export is an open feature request (huggingface/optimum#1897), not
   solid official support as of this check. Exporting it would mean a
   fragile, hand-rolled custom path for a signal that isn't even the
   dominant cost.
4. DINOv2 (style, ~0.8s) has clean, low-risk `ORTModelForFeatureExtraction`
   support, but contributes the least of the three local signals — not
   worth the engineering time on its own once (2) and (3) are settled.

**Decision**: no ONNX export was implemented. The true bottleneck (face
detection) is already ONNX and immovable without touching task 5.2's
shipped detector config; the one remaining local model with real export
risk (SigLIP) isn't the dominant cost; the one with low export risk
(DINOv2) isn't either. Real measured total across repeated runs:
**5.6-7.1s for 12 candidates** — borderline against the <6s target, not
a confident pass. Left open for revisit once daily-pipeline latency is a
concrete product concern (Phase 7+), not silently marked "done and fast."

## Decision — 6.7 critic node (identity-only hard gate)

CLAUDE.md's literal spec marks a panel for retry when "all three fail
hard thresholds (identity < t_id OR alignment < t_al)" — i.e. two
independent hard gates. `nodes/critic.py` implements **identity-only**
as the hard gate; alignment is deliberately not given a fixed absolute
threshold at all.

This follows directly from the 6.3 finding above: SigLIP's absolute
confidence on this project's flat-ink style is very low across the
board (correct and shuffled captions alike scored ~1e-4 to 1e-10) — a
real domain-shift artifact of SigLIP being trained on photography, not
comic art. A fixed `t_al` in that range would either always pass or
always fail regardless of whether a candidate actually matches its
intended scene, so it would gate nothing meaningful. What 6.3 *did*
verify reliably is alignment's **relative ranking** (correct scores
higher than shuffled 95% of the time) — that signal is still used, via
the reward head's weighted sum (task 6.5), just not as an independent
pass/fail cutoff.

`IDENTITY_THRESHOLD = 0.55` sits between 6.1's own real measured mean
negative score (0.367, a genuinely different character) and mean
positive score (0.738, the approved character) — a candidate below it
looks more like the wrong character than the right one.

**Retry/fallback mechanics**: up to `MAX_RETRIES = 2` regenerations per
panel via an injected `regenerate(panel_id)` callback (no `graph.py`
exists yet — task 7.4 — so this node is built and tested standalone,
matching the pattern `nodes/generate.py` already established). If every
retry is exhausted with no passing candidate, the node falls back to
the best-scoring candidate available (by reward, not identity alone)
and appends a descriptive string to `DayState.errors` — a panel is
never silently dropped. A real bug caught during self-review before
any test was written: a retry discards its failed batch entirely
(nothing keeps rejected candidates around, same as `nodes/generate.py`),
so the function must return whichever batch was actually chosen from —
the original candidates, or the last retry's fresh batch — not just the
single winning `Candidate`; returning only the winner would silently
drop it from `DayState.candidates` on any panel that needed a retry,
since it never existed in the caller's original list. Fixed before
writing tests, then covered by a dedicated regression test.

**Real verification (2026-09-15)**: 7 unit tests in
`tests/unit/llm/test_critic_node.py` — immediate pass without
retrying, highest-reward candidate wins among passers, all-retries-
exhausted falls back with an error appended, a late pass on the final
retry stops early with no error, the retry-batch-replacement bug above,
and `DayState.errors` accumulation (both panel groups and pre-existing
errors preserved). `ruff check`/`ruff format`/`mypy --strict` clean;
full suite green.

## Decision — 6.8 anti-hacking guards

CLAUDE.md names three guards. One was already satisfied before 6.8
formally started; the other two are new, in `critic/guards.py`.

**Cap identity weight**: already done by 6.5's `reward_head.py` —
`_clip01` clips every signal to `[0, 1]` before weighting, so no single
signal (identity or otherwise) can exceed its own weight's maximum
share even given a buggy or adversarial score outside `[0, 1]`. Covered
by `test_out_of_range_signal_is_clipped_not_allowed_to_dominate`
(written alongside 6.5, before 6.8 existed as a task). No new code
needed; this is the cross-reference CLAUDE.md's task list asks for.

**Framing-diversity bonus from script**: identity naturally rewards
large, clear faces, which a close framing structurally provides more of
than a wide one. Left unguarded, this is a real reward-hacking risk —
across 6.7's retries, the critic could implicitly push every panel
toward tight face crops, eroding the framing variety the script agent
deliberately assigns per panel (task 3.1's "at least one wide and one
close"). `framing_diversity_bonus` adds a small, fixed bonus
(`wide: 0.05, medium: 0.02, close: 0.0`) looked up from
`state.script.panels[panel_id].framing` in `nodes/critic.py`, applied on
top of the base reward-head score. When no script is wired (6.7's own
standalone tests), the bonus is `0.0` for every candidate — a no-op,
not a guess.

**OCR text penalty**: the style card (task 3.5) has no in-panel
lettering — dialogue is composed separately as captions/bubbles (task
7.1) — and garbled baked-in text is a well-known diffusion artifact none
of 6.1-6.4's signals would catch. **Package-name correction**: CLAUDE.md
names `rapidocr-onnx`, which does not exist on PyPI — verified live
against RapidOCR's current docs and GitHub before adding any dependency
(same discipline as 6.3's SigLIP/sentencepiece checks). The real,
current, maintained ONNX-backed package is `rapidocr`, with
`onnxruntime` installed separately by design as its inference backend.

**Real bug caught by testing live rather than trusting the docs**:
RapidOCR's own documentation describes an empty `boxes` array when no
text is detected. Running it for real on a blank image and a real
text image (2026-09-15) showed the actual behavior is `result.boxes is
None` in that case, not an empty array — `has_detected_text` handles
this with a defensive `getattr`, fixed before writing any unit test
against it (the unit tests then mock the engine, so this real finding
is preserved as an explicit regression test, not just a comment).

`has_detected_text`/`text_penalty`/`framing_diversity_bonus` compose
into `guarded_reward`, applied in `nodes/critic.py`'s `_reward()` on top
of the unmodified 6.5 reward-head score; `reward_head.py` itself is
untouched, keeping task 6.5's own accept line ("deterministic weighted
combo") intact as a separately-testable unit.

**Real verification (2026-09-15)**: 11 tests in
`tests/unit/llm/test_critic_guards.py` (the `None`-vs-empty-array bug,
bonus/penalty value checks, `guarded_reward` composition and
non-negativity) plus 4 integration tests in
`tests/unit/llm/test_critic_node.py` (a wide-framing bonus breaking an
exact tie, no-script-means-no-bonus, a detected-text penalty flipping a
ranking, and confirming the bonus values are sourced from `guards.py`
rather than duplicated). `ruff check`/`ruff format`/`mypy --strict`
clean; full suite (363 tests, verified via `pytest --collect-only`) green.

## Consequences

- `critic/identity.py`'s weighting (0.7 checklist / 0.3 ArcFace) is now
  the reference point for tasks 6.4–6.5's own thresholds, appended below
  as each lands.
- Task 6.6's <6s target is not reliably met today. Any future latency
  work should target face detection first (the real dominant cost, not
  currently one of CLAUDE.md's named four models) rather than the
  originally-named style/alignment models, which the real numbers show
  are not where the time actually goes.
- Every panel-candidate identity score costs one real VLM checklist call
  — a real, ongoing per-panel cost this pipeline now carries (already
  true since Phase 5's master/sheet ranking; task 6.1 extends the same
  cost structure to the daily generation path). Style scoring (6.2) adds
  no comparable per-call cost — it's a local CPU model.
- `ml/goldens/style/` (and `ml/goldens/identity/`, task 6.1's own
  literal-spec golden) remain real, deliberate gaps — both directories
  exist but are empty. A future pass should seed them from real chosen-
  panel history once enough exists, rather than leave the reg-set/raw-
  photo substitution as a permanent stand-in.
- `services/worker` now depends on `rapidocr`/`onnxruntime` (task 6.8).
  RapidOCR's own model files download and cache under the shared `.venv`
  on first real use (not committed to the repo) — the same pattern as
  InsightFace's buffalo_l models (task 5.2), not a new kind of gap.
- `nodes/critic.py`'s retry/fallback logic (6.7) and anti-hacking guards
  (6.8) are both built and tested standalone against an injected
  `regenerate` callback — no `graph.py` (task 7.4) exists yet to wire
  the real retry edge back to the prompt writer. Phase 7 should treat
  this node's existing test suite as the contract that wiring must not
  break, rather than re-deriving its retry/threshold/guard behavior.
- Gate 6 is now closed: all of 6.1-6.8 are real, tested, and numbered in
  this ADR. The one open caveat carried forward is 6.6's borderline
  latency number (previous bullet) — a known, documented gap, not a
  silent one.
