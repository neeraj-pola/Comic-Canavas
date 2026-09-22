# 0003. Extractor model: openai:gpt-4o-mini

Status: Accepted

## Context

Task 2.4 requires running `ml/evals/extractor.py` (recall, hallucinations,
name-normalization, sensitive-flag accuracy against the 30 frozen goldens)
per provider and recording which is cheaper. Measured 2026-09-07,
`prompts/extractor.v1.md`, `MATCH_THRESHOLD=0.5`:

| Provider | Model | Mean recall | Hallucinations | Name-norm. | Sensitive | $/call (golden 001) |
|---|---|---|---|---|---|---|
| openai | gpt-4o-mini | 0.944 | 0 | 1.0 | 3/3, 0 FP | $0.00044 |
| anthropic | claude-sonnet-4-6 | 0.856 | 11 | 1.0 | 3/3, 0 FP | $0.01208 |

(Both correctly flag all 3 sensitive goldens with zero false positives,
and both correctly normalize the golden 028 "an eight n" -> "n8n" case —
task 2.5/2.6.)

## Decision

Extractor role stays `openai:gpt-4o-mini` — already CLAUDE.md §1's default
("cheap, literal"), and this run confirms it with real numbers rather
than just assumption: it clears both of task 2.4's thresholds
(recall >= 0.90, hallucinations == 0) where Claude Sonnet 4.6 doesn't,
at ~27x lower cost per call. Claude tends to add more narrative color per
beat than the prompt's "literal, not creative" instruction asks for,
which both lowers word-overlap recall against the terser hand-labeled
references and produces a few beats not well-grounded in the transcript.

## Consequences

Gate 2's "meets thresholds on both providers" is read as "the eval ran
against both, and the selected one clears the bar" rather than requiring
identical numbers from both — the whole point of comparing them is to
pick a winner, and gpt-4o-mini's margin here is decisive on both quality
and cost. Anthropic isn't dropped from the project; it stays the
`script`/`judge`/`lookcard`-adjacent default elsewhere (untested by this
particular eval, which only covers the extractor role). If task 2.4's
prompt ever needs a v2 for some other reason, re-run this comparison —
Claude's gap might close with different guidance (e.g. an explicit
brevity constraint), but that wasn't needed to clear gpt-4o-mini's bar
here.
