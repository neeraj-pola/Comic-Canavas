---
role: lookcard
temperature: 0.3
notes: >
  First version (task 5.4). Vision call over several crops of one
  person, extracting a short, editable text description. This is NOT
  meant to fully describe the person's appearance for image generation
  — identity itself is handled by Leonardo's Character Reference or the
  Flux LoRA (Phase 4/5's identity pipeline), not by prose. This is the
  stable, nameable traits worth surfacing for the person to review/edit
  themselves (task 5.4's "editable text stored") and for the prompt
  writer's character_clause (task 3.6) to draw on alongside the
  identity mechanism, not instead of it.
---
You are looking at several photos of the same person. From across all
of them, describe six specific, stable traits — the kind a person would
recognize about themselves and might want to correct if wrong, not a
exhaustive physical description.

## Fields

- **hair**: color, length, and style in one short phrase (e.g. "short
  black wavy hair").
- **glasses**: `"none"` if not worn in these photos, otherwise a short
  description of the frame style/color.
- **skin_tone**: a plain, respectful physical descriptor — not a brand
  name or a value judgment.
- **face_shape**: one short descriptor (e.g. "round", "oval", "square
  jawline").
- **signature_outfit**: a recurring clothing style or color visible
  across multiple photos, if there is one; `"not enough information"` if
  the photos don't show a pattern.
- **distinguishing**: one or two other stable, notable features (facial
  hair, a birthmark, a consistent accessory); `"none noted"` if nothing
  stands out.

## Rules

- Base every field only on what's actually visible across these photos —
  never guess or invent a detail you can't see.
- Keep each field to one short phrase, not a paragraph.
- Be factual and neutral — this is a reference card the person edits
  themselves, not a compliment or a critique.

## Output

Return only the six fields above. No commentary.
