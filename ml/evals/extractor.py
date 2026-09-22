"""Extractor eval: beat recall, hallucination count, and
name-normalization accuracy against the frozen goldens in
`ml/goldens/transcripts/`, per provider.

The scoring functions (`load_goldens`, `score_case`) are pure and
network-free, covered by `tests/unit/ml/test_extractor_eval.py`. Only
`run_eval`/`main` need a real `LLM_EXTRACTOR` provider and key —
`make eval`'s job, not `make test`'s.

Run (from anywhere in the repo): `uv run python ml/evals/extractor.py` — a
plain script run, not `-m`, since this file isn't inside a package that
needs to be importable as one; it bootstraps `services/worker` onto
`sys.path` itself (below) so `app.*` resolves regardless of cwd.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from app.llm.routing import resolve  # noqa: E402
from app.memory import InMemoryMemoryStore  # noqa: E402
from app.nodes.beats import build_messages  # noqa: E402
from app.prompts.loader import load_prompt  # noqa: E402
from contracts import BeatSheet  # noqa: E402

GOLDENS_DIR = _REPO_ROOT / "ml" / "goldens" / "transcripts"
REPORTS_DIR = Path(__file__).parent / "reports"

# A predicted beat's event fuzzy-matching an expected beat's event at or
# above this ratio counts as recalled; below it, the expected beat is a
# miss and, symmetrically, an unmatched predicted beat is a hallucination.
#
# Word-overlap "containment" (shared significant words / the shorter side's
# word count), not difflib's character-sequence ratio: an LLM's own
# phrasing of the same event is often nothing like a character match to a
# hand-labeled reference even when it's a faithful paraphrase (e.g. "Went
# to the gym before work despite feeling tired" vs "Made it to the gym
# before work for once, felt good even though I was dragging the whole
# time" — same beat, ~0.53 character ratio). Containment (rather than
# Jaccard, overlap / union) also means a longer, more detailed prediction
# covering every word of a terser expected beat still scores 1.0, which
# matters since models are typically more verbose than a hand-labeled
# reference.
MATCH_THRESHOLD = 0.5

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "with", "despite", "although", "while", "after", "before", "during",
    "i", "my", "me", "was", "were", "is", "are", "felt", "feeling", "that",
    "this", "it", "its", "their", "them", "he", "she", "they", "we", "you",
    "your", "once", "even", "though", "so", "as", "by", "from", "up", "out",
}  # fmt: skip


def _stem(word: str) -> str:
    """Crude suffix-stripping — no dependency, just enough to stop "work" vs
    "worked" or "volunteer" vs "volunteering" costing a real content-word
    match."""
    for suffix in ("ing", "edly", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS and len(w) > 2}


@dataclass
class GoldenCase:
    id: str
    category: str
    known_people: list[str]
    known_places: list[str]
    known_vocab: list[str]
    transcript: str
    expected: BeatSheet


@dataclass
class CaseScore:
    id: str
    category: str
    recall: float
    hallucinations: int
    name_normalization_correct: int
    name_normalization_total: int
    sensitive_expected: bool
    sensitive_predicted: bool


@dataclass
class EvalReport:
    provider: str
    model: str
    prompt_version: str
    cases: list[CaseScore] = field(default_factory=list)

    @property
    def mean_recall(self) -> float:
        return sum(c.recall for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def total_hallucinations(self) -> int:
        return sum(c.hallucinations for c in self.cases)

    @property
    def name_normalization_accuracy(self) -> float | None:
        total = sum(c.name_normalization_total for c in self.cases)
        if total == 0:
            return None
        correct = sum(c.name_normalization_correct for c in self.cases)
        return correct / total

    @property
    def sensitive_recall(self) -> tuple[int, int]:
        """(flagged, expected) among goldens where sensitive is expected."""
        relevant = [c for c in self.cases if c.sensitive_expected]
        return sum(1 for c in relevant if c.sensitive_predicted), len(relevant)

    @property
    def sensitive_false_positives(self) -> int:
        return sum(1 for c in self.cases if c.sensitive_predicted and not c.sensitive_expected)

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "mean_recall": self.mean_recall,
            "total_hallucinations": self.total_hallucinations,
            "name_normalization_accuracy": self.name_normalization_accuracy,
            "sensitive_flagged_of_expected": list(self.sensitive_recall),
            "sensitive_false_positives": self.sensitive_false_positives,
            "cases": [
                {
                    "id": c.id,
                    "category": c.category,
                    "recall": c.recall,
                    "hallucinations": c.hallucinations,
                    "sensitive_expected": c.sensitive_expected,
                    "sensitive_predicted": c.sensitive_predicted,
                }
                for c in self.cases
            ],
        }


def load_goldens(directory: Path = GOLDENS_DIR) -> list[GoldenCase]:
    cases = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        cases.append(
            GoldenCase(
                id=raw["id"],
                category=raw["category"],
                known_people=raw["known_people"],
                known_places=raw["known_places"],
                known_vocab=raw["known_vocab"],
                transcript=raw["transcript"],
                expected=BeatSheet.model_validate(raw["expected"]),
            )
        )
    return cases


def _similar(a: str, b: str) -> float:
    wa, wb = _significant_words(a), _significant_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def score_case(golden: GoldenCase, predicted: BeatSheet) -> CaseScore:
    expected_events = [b.event for b in golden.expected.beats]
    predicted_events = [b.event for b in predicted.beats]

    matched_expected = 0
    for expected_event in expected_events:
        if any(_similar(expected_event, p) >= MATCH_THRESHOLD for p in predicted_events):
            matched_expected += 1
    recall = matched_expected / len(expected_events) if expected_events else 1.0

    # Hallucination: not "doesn't word-match a hand-labeled beat" — a
    # golden's expected beats sometimes deliberately merge several of the
    # transcript's sub-events into one terse label, so a model that instead
    # reports a couple of those sub-events as their own faithful beats
    # isn't fabricating anything, just choosing a different, still-correct
    # granularity. Checked against the transcript itself instead: since
    # `_similar` is
    # containment over the *shorter* side's words, and the predicted beat
    # is almost always shorter than the whole transcript, this asks "how
    # much of what this beat claims actually appears in the transcript" —
    # the real definition of "traceable to something the speaker said"
    # (extractor.v1.md) — regardless of how the reference happened to
    # summarize it.
    hallucinations = sum(
        1
        for predicted_event in predicted_events
        if _similar(predicted_event, golden.transcript) < MATCH_THRESHOLD
    )

    # Name-normalization accuracy: of the known_people this golden actually
    # exercises (i.e. mentioned in the expected output), how many appear
    # with their known_people spelling somewhere in the predicted output.
    predicted_names = set(predicted.people_mentioned)
    for beat in predicted.beats:
        predicted_names.update(beat.people)
    relevant_known = [p for p in golden.known_people if p in golden.expected.people_mentioned]
    name_correct = sum(1 for p in relevant_known if p in predicted_names)

    return CaseScore(
        id=golden.id,
        category=golden.category,
        recall=recall,
        hallucinations=hallucinations,
        name_normalization_correct=name_correct,
        name_normalization_total=len(relevant_known),
        sensitive_expected="sensitive" in golden.expected.flags,
        sensitive_predicted="sensitive" in predicted.flags,
    )


async def run_eval(goldens: list[GoldenCase]) -> EvalReport:
    provider, model = resolve("extractor")
    prompt = load_prompt("extractor")
    report = EvalReport(provider=provider.name, model=model, prompt_version=prompt.versioned_name)

    for golden in goldens:
        memory = InMemoryMemoryStore()
        memory.seed(
            "eval",
            people=golden.known_people,
            places=golden.known_places,
            vocab=golden.known_vocab,
        )
        messages = build_messages(
            prompt_text=prompt.text,
            transcript=golden.transcript,
            date_iso=golden.expected.date.isoformat(),
            known_people=memory.known_people("eval"),
            known_places=memory.known_places("eval"),
            known_vocab=memory.known_vocab("eval"),
        )
        predicted, _ = await provider.structured(
            messages, BeatSheet, model=model, temperature=prompt.temperature
        )
        report.cases.append(score_case(golden, predicted))

    return report


def main() -> None:
    goldens = load_goldens()
    report = asyncio.run(run_eval(goldens))

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"extractor_{report.provider}_{report.model.replace('/', '_')}.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    print(f"provider={report.provider} model={report.model} prompt={report.prompt_version}")
    print(f"mean_recall={report.mean_recall:.3f} (target >= 0.90)")
    print(f"total_hallucinations={report.total_hallucinations} (target == 0)")
    print(f"name_normalization_accuracy={report.name_normalization_accuracy}")
    flagged, expected = report.sensitive_recall
    print(
        f"sensitive: {flagged}/{expected} flagged, "
        f"{report.sensitive_false_positives} false positives"
    )
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
