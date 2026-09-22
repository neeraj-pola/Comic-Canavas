# 0013. Composer: strip/story rendering (Phase 7, tasks 7.1/7.2)

Status: Accepted (2026-09-15)

## Context

Task 7.1 asks for a Pillow-rendered `strip.png`/`story.png`/per-panel PNGs
with captions and a face-avoiding speech bubble; task 7.2 requires the
bubble to never overlap a face. Building this raised four real decisions
not settled elsewhere in CLAUDE.md.

## Decision — reading candidate bytes back from Storage

`DayState` only ever carries `Candidate.url` (needed for task 7.6's
idempotency: a resumed job can't rely on bytes still sitting in the
process that generated them). `Storage` had no read-back method. Rather
than add a second `Candidate.key` field that could drift out of sync
with `url`, `Storage` gained `get_object_by_url(url) -> bytes`, which
each backend implements by reversing its own `get_url` format
(`LocalFileStorage` strips its `base_url`/media prefix; `R2Storage`
strips its `public_base` prefix). `url` stays the single source of
truth for "where is this object."

## Decision — vendored variable fonts, not per-weight static files

CLAUDE.md names "Inter 600" for the caption bar. Google Fonts' current
mirror ships Inter and JetBrains Mono only as variable fonts (no static
weight instances). Verified live that Pillow 12's
`set_variation_by_axes` genuinely changes rendered weight (measured: a
700-weight render has 56% more dark pixels than 400-weight at the same
text and size) before committing to this path — one variable `.ttf` per
family, vendored under `services/worker/fonts/` with its `OFL.txt`,
covers every weight this project will ever need.

## Decision — 1x3 grid for the 3-panel case

CLAUDE.md names only "2x2 strip (or 1x2 quiet day)"; `Script.panels` is
2-4 generally (task 3.1). A single row of 3 is the natural shape between
the two named ones — implemented, not left as a crash.

## Decision — story.png letterboxes, doesn't crop

First real render (real Gate 5 candidate images) showed `story.png`'s
short, wide per-panel bands center-cropping candidate images down to
mostly-just-a-face — visually confirmed, not assumed. Fixed by
letterboxing (scale to fit, pad with the field color) instead of
cropping for the story format only; the square strip grid still
center-crops, where it doesn't lose meaningful content. `caption_a` (not
`caption_b`) is the caption baked into both — `caption_b` is reserved
for a future caption A/B loop (Phase 9), matching the live dashboard's
single displayed caption.

## Decision — Bangers for in-panel comic content, not Inter

Live user feedback on the first real render (real Gate 5 photos, shown
directly): Inter — correct for the app's own UI chrome (CLAUDE.md §10's
"Inter + JetBrains Mono") — read as a plain sans-serif for the actual
comic captions and speech bubbles, not "comic" at all. Fixed by adding
`Bangers` (SIL OFL, github.com/google/fonts/ofl/bangers), a genuine
comic-lettering display face, verified available as a real static TTF
before vendoring it. Scope is deliberately narrow: `_draw_caption_bar`/
`_draw_bubble` (the actual in-panel comic content) now render through
`_comic()`, uppercased (Bangers is an all-caps display face — the same
convention real comic lettering uses); the date header and mood pill
(surrounding UI chrome baked into the same image) stay in Inter/
JetBrains Mono, matching the app's own decided typography there. This
is a real, deliberate exception to CLAUDE.md §10's typography decision
for the composer's in-panel text specifically — comics conventionally
use dedicated lettering fonts for dialogue/captions, distinct from a
product's UI font, and the live-rendered result confirmed this reads as
intended once applied.

**Real process lesson, not a code bug**: the render that prompted this
feedback also showed a visibly inconsistent character and a caption
promising coffee over an unrelated image — traced to the demo script
picking the first PNG *alphabetically* out of `.data/candidates/panel-2/`,
which silently grabbed a leftover file from an unrelated earlier test
(a UUID-named artifact from a Sep 9 Leonardo/mock run) instead of the
real, task-5.13-verified consistency-tested image. Fixed by correlating
file mtimes against `ml/evals/reports/flux_kontext_4panel_2026-09-15.json`'s
own write time to identify the actual 4 images that eval scored — not a
`compose.py` defect, but a reminder that demo/reference assets need
their provenance checked before being trusted, the same discipline
already applied to third-party APIs and libraries throughout this
project.

## Verification

Real: 64 tests (`tests/unit/llm/test_compose.py`) including a 50-case
randomized bubble-avoidance sweep (task 7.2's literal accept line) with
a documented, provable zero-overlap bound (see `compose.py`'s module
docstring), and a pixel-diff test against a committed golden render
(`tests/golden/compose_strip_golden.png`, tolerance 0.5 mean abs
per-channel diff — a same-environment re-render measures exactly 0.0;
a real content change measures ~0.95, verified by a dedicated
"golden comparison actually discriminates" test so the tolerance check
can't silently always pass). Also visually inspected both `strip.png`
and `story.png` rendered from real Gate 5 candidate images before
writing any test — twice: once before the font/image-selection fixes,
once after, confirming the corrected render actually addresses the real
feedback (consistent character, coffee visible, comic lettering). Golden
regenerated deliberately after the font change (CLAUDE.md §0.1's
"golden sets change only via a PR titled `golden:` with a reason,"
applied here to this composer-specific golden the same way). `ruff`/
`mypy --strict` clean; full suite (447 tests) green.

## Consequences

- Any future composer change must keep the golden render's fixture
  (`tests/unit/llm/test_compose.py`'s `_fixture_*` helpers) in sync with
  `tests/golden/compose_strip_golden.png`, or regenerate the golden
  deliberately (mirroring `ml/goldens/`'s own "frozen, PR titled `golden:`"
  convention from CLAUDE.md §0.1, applied here for the same reason).
- `compose_day` (the pipeline-node entry point) is exercised directly
  today, the same pattern `nodes/generate.py` and `nodes/critic.py`
  established before `graph.py` (task 7.4) exists.
