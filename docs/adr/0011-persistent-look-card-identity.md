# 0011. Persistent look-card traits, deterministic character_clause

Status: Accepted

## Context

Diagnosed 2026-09-09 while investigating identity-fidelity test renders
(task 5.7): comic-styled test generations frequently rendered the wrong
gender, a near-bald head, and a more child-like/fuller face than the
source photos. Root cause was structural, not a tuning problem — traced
by actually inspecting the running pipeline's data, not just its code:

- `LookCard` (task 5.4) was live-verified but never persisted —
  `IdentityModel` had no field for it, so the vision call's output was
  used once and discarded.
- `nodes/prompts.py`'s `character_clause` was entirely LLM-authored
  (`promptwriter.v1.md`) and never drew on a look card at all — there
  was no wiring between the two, because task 3.6 was built before task
  5.4 (the look card) existed.
- Result: no panel prompt, ever, described the person's actual gender,
  hair, glasses, or features. Generation relied entirely on the identity
  mechanism (Flux LoRA / Leonardo Character Reference) to carry facts
  that mechanism is comparatively weak at, especially a 12-photo/1000-
  step LoRA (see ADR 0010's numbers and task 5.6's note).
- Separately, every LoRA training caption used a generic, genderless
  `"a photo of a person"` — non-standard for identity LoRA training
  (Dreambooth/LoRA convention anchors the trigger token to the actual
  class noun, "man"/"woman", not "person") and additional signal loss
  on top of the above.

## Decision

1. `LookCard` moves to `packages/contracts` (a real module boundary
   between node 0 and node 5) and gains `gender_term` — deliberately
   *not* vision-inferred (guessing gender from photos risks
   misgendering and isn't more reliable than the person stating it), so
   it's a plain caller-supplied field, never part of the vision call's
   own structured-output schema.
2. `IdentityModel` persists `look_card` for real.
3. `nodes/prompts.py`'s `character_clause` is no longer LLM-authored.
   It's built deterministically in code from the look card
   (`_describe_look_card`: gender, hair, glasses, face shape,
   distinguishing marks) plus `[IDENTITY]`. The LLM call
   (`PanelClauses`, `promptwriter.v2.md`) now only authors
   `character_expression_clause` (expression + a per-panel style rule
   like outfit continuity) and `environment_clause`.
4. LoRA training captions (`ml/identity/lora_dataset.py`) gain a fixed,
   caller-supplied prefix built from `gender_term` + the same stable
   look-card traits (hair, glasses) — fixed across every caption, same
   as the trigger token, so it doesn't reintroduce the "two competing
   constant signals" problem `trainingcaption.v1.md` warns against
   (only the per-photo scene description still varies per photo).

## Consequences

Stable identity traits now show up in every panel prompt and every
training caption the same way, every time — a code guarantee, not
something an LLM has to remember to restate from memory each call. This
doesn't replace the identity mechanism (LoRA/Character Reference), which
still carries the facial-structure "does this look like the same
person" signal words can't — it removes a second, avoidable source of
drift stacked on top of it. Re-test task 5.7's identity render once a
retrained LoRA (ADR 0010's step-count follow-up) and a real `LookCard`
with `gender_term` set are both in place, to separate how much each fix
independently helped.
