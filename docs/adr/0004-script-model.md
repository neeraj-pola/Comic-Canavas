# 0004. Script model: anthropic:claude-sonnet-4-6

Status: Accepted

## Context

Task 3.4 requires running `ml/evals/script.py` (constraint compliance,
mean caption length, LLM-judge pairwise win-rate vs the 15 frozen
`ml/goldens/scripts/` references) per provider. Measured 2026-09-07,
`prompts/script.v3.md`, judge role pinned to `anthropic:claude-sonnet-4-6`
for both runs so the judge itself doesn't vary between comparisons:

| Provider | Model | Mean compliance | Mean caption length | Judge win-rate |
|---|---|---|---|---|
| anthropic | claude-sonnet-4-6 | 1.000 | 43.5 chars | 0.933 |
| openai | gpt-4o | 1.000 | 31.9 chars | 0.100 |

(Compliance = beat_ids_valid && cast_matches_beats && framing_diverse,
averaged over all 15 goldens; both clear task 3.4/Gate 3's >= 0.98 bar
via `nodes/script.py`'s real framing-diversity retry — see the `script.v2`/
`script.v3` prompt fixes and the `run_eval` fix in the same commit series
that made this comparison exercise the actual node instead of a
reimplemented, retry-less call.)

## Decision

Script role stays `anthropic:claude-sonnet-4-6` — already CLAUDE.md §1's
default. The gap here is decisive and one-sided: Claude's scripts win or
tie the judge 93.3% of the time against hand-written reference scripts;
GPT-4o's win only 10% of the time — losing to the same references nine
times out of ten. Skimming both providers' raw output, GPT-4o's panels
are compliant (valid beat_ids, matching cast, diverse framing) but
noticeably flatter: shorter, more generic captions (31.9 vs 43.5 mean
chars) that state what happened rather than finding a specific, textured
detail the way the references and Claude's output both do. Compliance
alone doesn't catch this — it's exactly what the judge step is for.

## Consequences

Gate 3's "script constraint compliance >= 98%" is met by both providers,
so the deciding factor is judge win-rate, per task 3.4's framing (compare
providers, pick a winner) rather than compliance being the only bar.
GPT-4o isn't dropped from the project; it stays the `extractor`/`prompts`/
`lookcard` default elsewhere (ADR 0003; untested here, which only covers
the `script` role). If `script.v3` is revised for some other reason,
re-run this comparison — a more directive prompt (explicit "favor a
specific sensory or emotional detail over a generic summary") might close
GPT-4o's gap, but that wasn't needed to clear Claude's bar here.
