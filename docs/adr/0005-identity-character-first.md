# 0005. Identity is a designed character, not a photo-conditioned face

Status: Accepted

## Context

Phase 5's original design tried to carry a person's identity through a
photo-conditioned generation mechanism (a Flux LoRA trained on their real
photos, or Leonardo's Character Reference) directly into a stylized
comic panel. Measured result: the v1 photo LoRA scored mean ArcFace
identity **0.37 against a 0.45 target** (task 5.7's old accept line,
before this ADR) — a real, quantified failure, not a subjective one.

Live testing this session, across every identity mechanism actually
tried, found the same pattern: a trained Flux LoRA (v1 and v2, the
latter with a boosted alpha and 2000 steps), Leonardo Character
Reference, and PuLID-Flux (a zero-shot face-embedding conditioning
model) all preserved coarse traits (age, gender, hair color, glasses)
reasonably well but lost the person's exact facial geometry the moment
real comic stylization was applied. Two prior fixes (ADR 0010's GPU
switch, ADR 0011's persistent look-card traits) were both real,
necessary corrections but neither fixed this — ruling out "insufficient
training" or "missing prompt data" as the root cause. The problem is
structural: asking a generator to hold a specific face's geometry
*through* a stylization transform is a harder, less reliable task than
either photo-realism or stylization alone.

## Decision

Identity becomes a **designed comic character**, approved by the person
once, rather than a photo pushed through stylization on every panel.
Concretely:

1. The person approves one **master design** (task 5.8) — a stylized
   portrait generated from their real photo, which *becomes* the
   reference for every future panel (and, eventually, task 5.11's
   character-sheet LoRA). Identity only needs to survive the
   photo-to-comic transform once, under human supervision with
   regeneration available, not silently on every generated panel.
2. The **feature checklist** (task 5.9) replaces ArcFace cosine
   similarity as the actual identity gate. ArcFace is built for
   photographic face recognition; live testing found it frequently
   can't even detect a face in a genuinely stylized comic render, let
   alone score its similarity meaningfully. "Does this drawn face show
   every feature of the approved character" is both more answerable by
   a vision-language model and more aligned with what actually matters
   for a comic diary — recognizability of the *design*, not
   photographic geometric identity.
3. Role changes to existing Phase 5 tasks: 5.3's ArcFace embedding
   becomes a weak ranking/tie-break signal (not the gate); 5.4's look
   card becomes the checklist's ground truth, gaining `mandatory`
   (which fields are pass/fail gates) and `edits` (a chip-edit audit
   trail); 5.5's Leonardo reference photos are superseded for actual
   generation once a master exists (re-uploaded from the master itself,
   not the raw intake photos); 5.6's photo LoRA is kept only as a
   documented baseline experiment, not the production path.

## Real backend found live (2026-09-14)

The plan going into this work assumed a canny-controlnet approach
(`fal-ai/flux-general` with an explicit ControlNet weights path) for
task 5.8's "structure-guided stylization." Live testing found a better
option: `fal-ai/flux-2/edit`, a multi-image *editing* model, given the
person's real face crop plus a project-wide style reference image
(`services/worker/app/prompts/style_reference.jpg`) — it edits the
actual photo's pixels toward the reference style, rather than
generating from a weaker conditioning signal. This beat every
from-scratch generator tried this session on both identity and style
fidelity, and is cheaper ($0.012/megapixel vs. $0.035 originally
estimated for the controlnet path).

One real, load-bearing trade-off found live: every mechanism tried this
session (including this one) shows some bias toward rendering rounder,
fuller faces than the source photo. Pushing the prompt hard against this
bias (repeated negative instructions) measurably pushes the model away
from applying the style transfer at all, reverting to near-monochrome
line art. `ml/identity/master.py`'s `build_master_prompt` states the
face-shape constraint once, plainly, rather than repeating it — the
balance that held both a narrower jaw and full color in live testing.
`GUIDANCE_SCALE=3.5` (not the endpoint's own default 2.5) was the value
that held that balance.

## Consequences

- Feature-checklist precision (task 5.9's literal accept line, ≥ 0.90 on
  mandatory features from a 40-crop labeled set) is now the quality bar
  this pipeline is measured against, not an ArcFace threshold.
- `ml/identity/train_lora.py`'s photo-LoRA trainer is kept as working
  code (task 5.11 may reuse it against a different dataset — a character
  sheet generated from the approved master, not raw photos) but is not
  on the critical path for a person's first strip.
- Gate 5 is not reachable from this ADR alone — tasks 5.10 through 5.18
  (character sheet, LoRA v2, reference-path generation, face refinement,
  the LoRA-vs-reference gate, recurring characters, the onboarding
  contract) remain open, deliberately out of scope for this pass.

## Update 2026-09-15: the identity gate (5.15) — reference path confirmed as the default

Tasks 5.10 (character sheet), 5.13 (reference-path generation), 5.11
(character LoRA), 5.12 (checkpoint selection), and 5.15 (the gate itself)
all completed this session, giving this ADR's central hypothesis — that
identity should live in a designed, human-approved character rather than
be re-learned by a trained model — its first real, direct test.

**5.13 (reference path, zero training)**: `images/flux_kontext.py`
generates daily panels by referencing the approved master + the shared
style reference directly (task 5.8's mechanism, reused for full scenes).
Real 4-panel eval: **mandatory checklist pass rate 1.000, mean face-crop
DINOv2-to-master similarity 0.816** — both comfortably clear the accept
line, with no training step at all. Getting here required two real fixes
during eval, not the generation approach itself: the eval's own DINOv2
metric initially compared whole frames (composition-mismatched against a
tight master headshot) rather than face crops, and one panel under a
strained expression genuinely lost the mustache until the identity-lock
prompt wording was strengthened to explicitly guard against that.

**5.11 (character LoRA, trained on the task 5.10 sheet)**: two real
training runs on Modal L40S, both technically completing without error
but **failing on the actual identity metric**:
- Run 1: 35 real sheet images + the full 150-image task 5.7 regularization
  set (`is_reg: true`). Verified against ai-toolkit's own
  `toolkit/data_loader.py` (a plain `ConcatDataset` + shuffle across all
  datasets, sampled proportional to image count, no reg/instance
  balancing) — the 150:35 ratio meant ~81% of the 2000 training steps
  touched a generic regularization image, not the actual character.
  Result: **0.00 mandatory-checklist pass rate at every one of 9
  checkpoints** (sample grids compared directly at steps 0/750/1250/1750/
  2000 for the same prompt/seed — visibly unchanged across the entire run).
  ~92 min, ~$3.
- Run 2: reg set capped to 35 (evenly sampled from the 150 for subject
  diversity), a standard ~1:1 reg:instance ratio. Real improvement in the
  composite score (0.31 → 0.46 from step 0 to step 2000) and partial
  trait recovery (glasses and mustache appeared in some samples, absent
  in run 1) — but **still 0.00 mandatory-checklist pass rate at every
  checkpoint**; hair color and face shape never converged to the training
  set. Took ~193 min (~$6, a real cost overrun against the ≤60min accept
  line) — Modal queue/runtime variance, not a config change.

**5.12 (checkpoint selection)**: scored all 9 checkpoints × 8 sample-grid
images (72 real images, reusing ai-toolkit's own training-time samples —
no new generation spend) with the checklist + face-crop DINOv2. Selection
correctly picked the actual best (step 2000, composite 0.458) over weaker
intermediate checkpoints — the automation itself works — but "best" here
still means 0% mandatory pass, not a usable result.

**5.15 (the gate)**: not run as a fresh 20-prompt head-to-head generation
— 5.12's 72-sample evidence already answered it decisively (LoRA best
case 0.00 mandatory pass vs. reference's 1.000), and spending further
real money on a live comparison that couldn't plausibly reverse so clear
a result was judged not worth it. **Decision: `reference` wins.**
`IdentityModel.generation_path = "reference"`, `lora_checkpoint` recorded
for provenance (`character_v1_me_v2_000002000`) but not promoted.

This is the third and fourth real, independent confirmation (after the
photo LoRA v1/v2 and Leonardo Character Reference/PuLID-Flux findings
that motivated this ADR in the first place) that a trained mechanism
underperforms a reference-based one for this pipeline. The character
LoRA experiment stays as documented, working code (task 9.5's Diffusion-
DPO learning track can still target it later, per CLAUDE.md's own
hybrid "production track / learning track" framing) — it is not deleted,
just not the default a real user's panels go through.

## Update 2026-09-15: face refinement (5.14) evaluated and not adopted

Task 5.14 assumed daily panels would need a post-generation face-quality
boost on wide/medium framing — a reasonable assumption under the
mechanisms this ADR already rejected (a trained LoRA or Leonardo's
Character Reference), which do plausibly render faces smaller/weaker at
wider framing. It does not hold for the winning reference path: real Gate
5 testing (9 of a planned 20 panels — the run was cut short by a real
fal.ai account balance lockout, not a code issue) found `flux_kontext`'s
identity-lock recipe already passes the mandatory checklist on **100% of
panels before any refinement**, across every framing tested including
wide and medium.

`images/refine_face.py` was built and tested (crop the detected face
region, upscale 2-3x, re-edit through the same master+style-locked
recipe, feather-composite back) and then evaluated live rather than
assumed to help. Of 7 refined wide/medium panels: **0 improved, 2
regressed** (one panel's checklist dropped from a perfect score to an
outright mandatory-feature failure), **5 were unchanged**. The mechanism
itself explains this: refinement re-runs the same stochastic editing
model a second, independent time on a crop that was usually already
good — a real chance to drift away from the locked identity (the exact
failure class this session already had to guard against once, in
`flux_kontext.py`'s own gaze/expression lock) for no measured benefit.

**Decision: face refinement is not part of the default pipeline.**
`refine_face.py` is kept as real, working, tested code (14 unit tests,
mypy --strict clean) — same treatment as the baseline photo-LoRA trainer
and Leonardo provider elsewhere in this ADR — available if a future
generation mechanism's own failure mode actually calls for it, but not
invoked today.

## Update 2026-09-15: the real 20-panel result, without refinement

Finished all 20 panels (11 remaining after the fal.ai balance top-up),
un-refined `flux_kontext` output directly. **Real result: 19/20 = 0.950
mandatory-checklist pass rate** — clears Gate 5's ≥0.90 bar. By framing:
medium 9/9 (1.00), close 5/5 (1.00), wide 5/6 (0.83).

The one failure (panel 19: a true wide establishing shot with the
character occupying a small fraction of the frame among trees, a path,
and other people) was visually confirmed, not just trusted from the
score — the mustache is genuinely too small to render reliably at that
scale. This is the expected trade-off CLAUDE.md's own style card already
names ("environments carry the detail budget; characters stay simple" at
wide framing), not a new bug, and the earlier refinement experiment
already showed post-hoc fixing doesn't reliably help this failure mode
either. One real, explainable outlier does not threaten a 95% result.

Gate 5's numeric bar is met for the owner. Not yet closed: "one side
character" (task 5.17, a recurring character) has not been attempted —
only the owner's own identity has run through this pipeline so far.
