# ml/goldens/transcripts/

Frozen per CLAUDE.md §0.1 — change only via a PR titled `golden:` with a
reason, and never to make an eval pass (iterate the prompt version instead,
per task 2.4).

30 files, `NNN_slug.json`, each:

```json
{
  "id": "001_...",
  "category": "real | synthetic | quiet | sensitive | jargon",
  "known_people": ["..."],
  "known_places": ["..."],
  "known_vocab": ["..."],
  "transcript": "...",
  "expected": { "...": "a full BeatSheet" }
}
```

`known_people`/`known_places`/`known_vocab` are the context `nodes/beats.py`
would have looked up from memory for this (synthetic) user at the time —
supplied here directly so a golden is reproducible without a real memory
store. `expected` is hand-labeled ground truth, not a real model output.

Breakdown (task 2.3 / CLAUDE.md §4): 10 `real` (naturalistic variety) +
20 `synthetic`, split `synthetic` (10, general variety) / `quiet` (4, few
or no events) / `sensitive` (3, grief/illness/breakup — must NOT be the
only thing that trips the `sensitive` flag; the 10 `real` + 10 `synthetic`
+ 4 `quiet` days cover ordinary bad days/stress on purpose, as the 0
false-positive half of task 2.5's accept line) / `jargon` (3, known_people
nicknames + known_vocab whisper-corruptions together, task 2.6).
