---
role: lookcard
temperature: 0.5
notes: >
  First version (task 5.6). One short, non-identity scene caption per
  LoRA training photo — reuses the "lookcard" role/provider (already
  vision-capable, task 5.4) rather than registering a whole new LLM_*
  role for one narrow purpose.

  Why non-identity matters here specifically: every training caption is
  prefixed with the same fixed trigger token by the caller, so the
  *only* thing constant across all captions is that token. If this
  prompt also described the person's appearance, the LoRA would have two
  competing "constant" signals to latch onto and could partially learn
  the wrong one (or blend them). Keeping this prompt strictly to
  background/pose/clothing — varying per photo — is what forces the
  model to associate the trigger token with identity specifically, not
  with anything else that happens to be visible.
---
Describe this photo's setting in one short, plain phrase: the
background/location, the person's pose or activity, and their clothing
if visible. Do not describe the person's face, hair, skin, or any other
physical feature — that's deliberately excluded, not an oversight.

Example: "outdoors near a lake, wearing a blue jacket, hands in pockets"

Return only that one phrase. No commentary, no punctuation at the end,
no mention of appearance.
