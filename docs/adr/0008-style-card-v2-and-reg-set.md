# 0008. Style card v2 and the regularization set

Status: Accepted

## Context

CLAUDE.md task 5.7 asks for a rewritten style card ("flat ink, bold even
outlines, flat cel shading, limited warm palette, halftone, detailed
environments, simple characters") and a ~150-image regularization set —
generic, identity-free comic-character renders used as the reference
corpus `ml/identity/style_score.py`'s DINOv2-small scorer compares
master-design candidates (task 5.8) against, to answer "does this read
as on-style" independent of who's in the picture.

Task 5.7's own text says the style-card decision should be recorded in
"ADR 0011" — but `docs/adr/0011-persistent-look-card-identity.md`
already exists (a different decision, from earlier the same week). ADR
numbers already spoken for elsewhere in CLAUDE.md: 0005 (this phase's
pivot, ADR 0005), 0006 (Gate 6, critic thresholds), 0007 (Gate 9,
preference optimization), 0012 (task 5.16's ZipLoRA experiment). 0008 is
the only free number — used here instead.

## Decision

**Style card v2** (`services/worker/app/prompts/style_card.yaml`,
`version: v2-2026-09-14`): replaced v1's loose hand-drawn ink line and
single mustard spot-accent with bold, even-weight outlines and flat cel
shading — a deliberate change, not a cosmetic one: v1's "loose, slightly
imperfect by hand" line style gave identity-conditioned generation (Flux
LoRA, Leonardo Character Reference) too much room to drift from render to
render, where bold closed-shape outlines and flat color fills are far
more reproducible under both reference-image editing (this phase's
approach) and a future LoRA. `style_phrase` is a new field: one canonical
style string every consumer (the regularization set, master-design
candidates, and eventually per-panel prompts) draws from verbatim.

**Regularization set** (`ml/identity/reg_set.py`): 150 images, `fal-ai/
flux/dev`, `image_size="square"` (512×512 — DINOv2-small ingests 224px,
so more resolution than that is wasted spend), 25 generic subjects
cycled across seeds, real cost **$1.3763** total (verified live
2026-09-14: a 5-image test batch cost $0.0459, in line with the
per-image estimate before committing to the full batch). The resulting
DINOv2-small embedding bank (`ml/identity/reg_embeddings/
comic_character_dinov2_small.npy`, 150×384 floats) has a mean top-10
self-similarity of **0.618** (min 0.304, max 0.784) — meaningfully
varied, not degenerate, confirming the bank captures a real, coherent
visual family rather than 150 near-duplicates or 150 unrelated images.

**Storage deviation from task 5.7's literal "committed via LFS or R2"**:
this repo has neither configured — no `.gitattributes`/Git LFS setup,
and R2 is Phase 12 by CLAUDE.md §0.1's own rule ("No deploys mid-build").
`ml/reg/` (the 150 raw images) is gitignored; only `manifest.json`
(recording every image's seed/prompt/subject, force-added past the
ignore) and the derived embedding bank are committed. The images
themselves are fully regenerable from the manifest if ever needed.

## Consequences

Task 5.7's accept line ("DINOv2 style reference set built from these")
is satisfied by the committed embedding bank. Any future style-card
change (a deliberate v3) invalidates this bank and requires regenerating
the regularization set under the new style — a real, non-trivial cost
(~$1.38 at these prices) to weigh before revising the style card lightly.
`IdentityModel.style_card_version` records which version a given
person's master design was made under, so a style-card revision doesn't
silently desync existing approved characters from what `master.py` would
produce today.
