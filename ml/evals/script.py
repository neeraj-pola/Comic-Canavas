"""Script eval: constraint compliance, mean caption length, and LLM-judge
pairwise win-rate against the frozen goldens in `ml/goldens/scripts/`, run
with a given script model.

`load_script_goldens`/`constraint_compliance` are pure and network-free,
covered by `tests/unit/ml/test_script_eval.py`. Only `run_eval`/`main`
need real `LLM_SCRIPT`/`LLM_JUDGE` providers and keys.

Run (from anywhere in the repo): `uv run python ml/evals/script.py` — see
`ml/evals/extractor.py`'s docstring for why this is a plain script run,
not `-m`.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from pydantic import BaseModel  # noqa: E402

from app.llm.base import Message  # noqa: E402
from app.llm.routing import resolve  # noqa: E402
from app.memory import InMemoryMemoryStore  # noqa: E402
from app.nodes.script import has_framing_diversity, write_script  # noqa: E402
from app.prompts.loader import load_prompt  # noqa: E402
from contracts import BeatSheet, DayState, Script  # noqa: E402

GOLDENS_DIR = _REPO_ROOT / "ml" / "goldens" / "scripts"
REPORTS_DIR = Path(__file__).parent / "reports"


class JudgeVerdict(BaseModel):
    winner: Literal["a", "b", "tie"]
    reasoning: str


@dataclass
class ScriptGolden:
    id: str
    source_transcript_id: str
    beat_sheet: BeatSheet
    reference_scripts: list[Script]


@dataclass
class ComplianceCheck:
    beat_ids_valid: bool
    cast_matches_beats: bool
    framing_diverse: bool

    @property
    def score(self) -> float:
        checks = [self.beat_ids_valid, self.cast_matches_beats, self.framing_diverse]
        return sum(checks) / len(checks)


@dataclass
class CaseResult:
    id: str
    compliance: ComplianceCheck
    mean_caption_length: float
    judge_results: list[Literal["win", "tie", "loss"]] = field(default_factory=list)


@dataclass
class ScriptEvalReport:
    script_provider: str
    script_model: str
    prompt_version: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def mean_compliance(self) -> float:
        return sum(c.compliance.score for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def mean_caption_length(self) -> float:
        lengths = [c.mean_caption_length for c in self.cases]
        return sum(lengths) / len(lengths) if lengths else 0.0

    @property
    def judge_win_rate(self) -> float:
        results = [r for c in self.cases for r in c.judge_results]
        if not results:
            return 0.0
        wins = sum(1 for r in results if r == "win")
        ties = sum(1 for r in results if r == "tie")
        return (wins + 0.5 * ties) / len(results)

    def to_dict(self) -> dict[str, object]:
        return {
            "script_provider": self.script_provider,
            "script_model": self.script_model,
            "prompt_version": self.prompt_version,
            "mean_compliance": self.mean_compliance,
            "mean_caption_length": self.mean_caption_length,
            "judge_win_rate": self.judge_win_rate,
            "cases": [
                {
                    "id": c.id,
                    "compliance": c.compliance.score,
                    "mean_caption_length": c.mean_caption_length,
                    "judge_results": c.judge_results,
                }
                for c in self.cases
            ],
        }


def load_script_goldens(directory: Path = GOLDENS_DIR) -> list[ScriptGolden]:
    goldens = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        goldens.append(
            ScriptGolden(
                id=raw["id"],
                source_transcript_id=raw["source_transcript_id"],
                beat_sheet=BeatSheet.model_validate(raw["beat_sheet"]),
                reference_scripts=[Script.model_validate(s) for s in raw["reference_scripts"]],
            )
        )
    return goldens


def constraint_compliance(beat_sheet: BeatSheet, script: Script) -> ComplianceCheck:
    beat_by_id = {b.id: b for b in beat_sheet.beats}
    beat_ids_valid = all(p.beat_id in beat_by_id for p in script.panels)
    cast_matches_beats = all(
        set(p.cast) == set(beat_by_id[p.beat_id].people)
        for p in script.panels
        if p.beat_id in beat_by_id
    )
    return ComplianceCheck(
        beat_ids_valid=beat_ids_valid,
        cast_matches_beats=cast_matches_beats,
        framing_diverse=has_framing_diversity(script) if len(script.panels) >= 2 else True,
    )


def mean_caption_length(script: Script) -> float:
    lengths = [len(p.caption_a) for p in script.panels] + [len(p.caption_b) for p in script.panels]
    return sum(lengths) / len(lengths) if lengths else 0.0


def _format_script_for_judge(beat_sheet: BeatSheet, script: Script) -> str:
    beats_by_id = {b.id: b for b in beat_sheet.beats}
    lines = [f"mood={script.mood} quiet_day={script.quiet_day}"]
    for p in script.panels:
        beat = beats_by_id.get(p.beat_id)
        lines.append(
            f"- beat: {beat.event if beat else '(unknown beat)'}\n"
            f"  framing={p.framing} expression={p.expression} action={p.action!r}\n"
            f"  caption_a={p.caption_a!r} caption_b={p.caption_b!r} bubble={p.bubble!r} "
            f"cast={p.cast}"
        )
    return "\n".join(lines)


async def judge_pairwise(
    beat_sheet: BeatSheet, candidate: Script, reference: Script
) -> Literal["win", "tie", "loss"]:
    """Judges `candidate` against `reference`, from the candidate's perspective."""
    prompt = load_prompt("judge")
    provider, model = resolve("judge")

    candidate_is_a = random.random() < 0.5
    script_a, script_b = (candidate, reference) if candidate_is_a else (reference, candidate)

    messages: list[Message] = [
        {"role": "system", "content": prompt.text},
        {
            "role": "user",
            "content": (
                f"BeatSheet:\n{beat_sheet.model_dump_json()}\n\n"
                f"Script A:\n{_format_script_for_judge(beat_sheet, script_a)}\n\n"
                f"Script B:\n{_format_script_for_judge(beat_sheet, script_b)}"
            ),
        },
    ]
    verdict, _ = await provider.structured(
        messages, JudgeVerdict, model=model, temperature=prompt.temperature
    )

    if verdict.winner == "tie":
        return "tie"
    candidate_won = (verdict.winner == "a") == candidate_is_a
    return "win" if candidate_won else "loss"


async def run_eval(goldens: list[ScriptGolden]) -> ScriptEvalReport:
    provider, model = resolve("script")
    prompt = load_prompt("script")
    report = ScriptEvalReport(
        script_provider=provider.name, script_model=model, prompt_version=prompt.versioned_name
    )

    for golden in goldens:
        # Go through the real write_script() node, not a re-implemented
        # provider.structured() call — the node's framing-diversity retry
        # is exactly what makes compliance representative of production
        # behavior; bypassing it here would score a weaker path than the
        # one that actually runs.
        state = DayState(
            job_id=f"eval-{golden.id}",
            user_id="eval",
            date=golden.beat_sheet.date,
            source="text",
            beats=golden.beat_sheet,
        )
        result_state = await write_script(state, memory=InMemoryMemoryStore(), humor_level=5)
        assert result_state.script is not None
        candidate = result_state.script

        case = CaseResult(
            id=golden.id,
            compliance=constraint_compliance(golden.beat_sheet, candidate),
            mean_caption_length=mean_caption_length(candidate),
        )
        for reference in golden.reference_scripts:
            case.judge_results.append(await judge_pairwise(golden.beat_sheet, candidate, reference))
        report.cases.append(case)

    return report


def main() -> None:
    goldens = load_script_goldens()
    report = asyncio.run(run_eval(goldens))

    REPORTS_DIR.mkdir(exist_ok=True)
    safe_model = report.script_model.replace("/", "_")
    out_path = REPORTS_DIR / f"script_{report.script_provider}_{safe_model}.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    print(
        f"provider={report.script_provider} model={report.script_model} "
        f"prompt={report.prompt_version}"
    )
    print(f"mean_compliance={report.mean_compliance:.3f} (target >= 0.98)")
    print(f"mean_caption_length={report.mean_caption_length:.1f} chars")
    print(f"judge_win_rate={report.judge_win_rate:.3f} (vs reference scripts, 0.5 = tied)")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
