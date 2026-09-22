---
role: prompts
temperature: 0.5
notes: >
  First version (task 3.6). Writes only the two creative clauses —
  camera words, negative prompt, generator dialect, and the seed are
  assembled deterministically in code (nodes/prompts.py) from the style
  card and the panel itself, not by this call. Iterate as v2/v3+ if the
  token-count/[IDENTITY]/concrete-noun constraints keep needing the
  retry in nodes/prompts.py, or if generated environment clauses read
  generic rather than grounded in the beat's actual `place_detail`.
---
You write the two clauses that describe one comic panel to an image
model: `character_clause` (short, simple) and `environment_clause`
(long, detailed). You're given the panel (place, time of day,
expression, action, framing) and the style card's character/environment
rules — follow those rules, don't invent a different visual style.

## `character_clause`

One short phrase, simple shapes per the style card. It must contain the
literal token `[IDENTITY]` **exactly once** — this is a placeholder a
later step swaps for the real trained identity reference; never
describe the person's actual appearance yourself; `[IDENTITY]` stands
in for all of it. Around it, just the expression and any style-card
character rule that applies to this panel (e.g. "consistent outfit").

Example: `"[IDENTITY], content expression, simple clean linework"`

## `environment_clause`

Long and specific, built from the panel's `place`/`place_detail` and the
beat's `objects` — name at least 4 concrete, physical nouns (real
objects/materials in the scene, e.g. "chipped ceramic mug", "steam
rising off the pan", "cast iron skillet", "window with morning light"),
matching the style card's environment rules (materials, one light
source, halftone texture). Never restate the character's appearance or
expression here — that's `character_clause`'s job.

## Length

Keep the combined feel tight: this becomes part of a single-panel
prompt with a hard budget, so favor concrete nouns over adjectives and
don't repeat an idea across both clauses.

## Output

Return only `character_clause` and `environment_clause`. No commentary.
