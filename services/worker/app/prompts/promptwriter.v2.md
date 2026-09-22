---
role: prompts
temperature: 0.5
notes: >
  v2 (2026-09-09), fixing a real bug found in v1: v1 asked the LLM to
  write the entire `character_clause`, including the literal
  `[IDENTITY]` placeholder and (implicitly, by never being told
  otherwise) nothing about the person's actual stable features — so
  nothing in the assembled prompt ever described the person's gender,
  hair, glasses, or other traits at all. `character_clause` is now built
  in code from the look card (`nodes/prompts.py`'s `_describe_look_card`)
  plus this call's much narrower output: just the expression and any
  panel-specific style rule (e.g. outfit continuity). This prompt no
  longer produces `character_clause`, and no longer needs the
  `[IDENTITY]`-placeholder instruction at all — see v1 for the prior
  version's framing if this needs a v3.
---
You write two things that describe one comic panel to an image model:
`character_expression_clause` (short) and `environment_clause` (long,
detailed). You're given the panel (place, time of day, expression,
action, framing) and the style card's character/environment rules —
follow those rules, don't invent a different visual style. The person's
stable features (gender, hair, glasses, face shape, distinguishing
marks) are handled elsewhere, in code — never mention them here.

## `character_expression_clause`

One short phrase: the panel's expression, plus any style-card character
rule that actually applies to this panel (e.g. "consistent mustard
sweater" for outfit continuity — you're given `signature_outfit` for
this). Do not describe hair, face, glasses, skin, or any other physical
feature — that's handled elsewhere, not your job here.

Example: `"a warm, restrained smile, wearing the same mustard sweater"`

## `environment_clause`

Long and specific, built from the panel's `place`/`place_detail` and the
beat's `objects` — name at least 4 concrete, physical nouns (real
objects/materials in the scene, e.g. "chipped ceramic mug", "steam
rising off the pan", "cast iron skillet", "window with morning light"),
matching the style card's environment rules (materials, one light
source, halftone texture). Never restate the character's appearance or
expression here — that's `character_expression_clause`'s job.

## Length

Keep the combined feel tight: this becomes part of a single-panel
prompt with a hard budget, so favor concrete nouns over adjectives and
don't repeat an idea across both clauses.

## Output

Return only `character_expression_clause` and `environment_clause`. No
commentary.
