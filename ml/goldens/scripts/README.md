# ml/goldens/scripts/

Frozen per CLAUDE.md §0.1 — change only via a PR titled `golden:` with a
reason, and never to make an eval pass (iterate the prompt version
instead, per task 3.4).

15 files, `NNN_slug.json`, each:

```json
{
  "id": "NNN_slug",
  "source_transcript_id": "the ml/goldens/transcripts/ golden this beat_sheet came from",
  "beat_sheet": { "...": "a full BeatSheet" },
  "reference_scripts": [ { "...": "a full Script" }, { "...": "a full Script" } ]
}
```

`beat_sheet` is copied verbatim from one of the 30 `ml/goldens/transcripts/`
goldens' `expected` field (reusing that frozen world rather than
authoring 15 more from scratch) — a representative mix of `real` /
`synthetic` / `quiet` / `sensitive` / `jargon` categories. `reference_scripts`
are two independently-plausible, hand-written `Script`s for the same
beat sheet (different panel selection and/or captions/framing), both
satisfying the same constraints a generated script must (2-4 panels,
caption/bubble lengths, at least one `wide` and one `close` framing,
`beat_id`s that exist in `beat_sheet`) — used by `ml/evals/script.py`
(task 3.4) as the LLM-judge's pairwise comparison targets and by
`nodes/script.py`'s constraint validation as a correctness check (task
3.3: "all 15 script goldens produce valid Script").

The two `quiet`-category goldens (012, 013) have only one beat each
(`too_short`) — both of that beat's reference-script panels reuse the
same `beat_id`, since a one-beat day still gets 2 panels (task 3.1) by
drawing two different visual moments out of that single beat, not by
inventing a second event.
