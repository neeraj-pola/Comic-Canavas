"""Feature-checklist eval: precision on mandatory features against a
hand-labeled set of face crops, per provider. Mirrors `ml/evals/
extractor.py`'s split: pure scoring (`load_goldens`, `score_case`) is
network-free and covered by `tests/unit/ml/test_checklist_eval.py`; only
`run_eval`/`main` need a real `LLM_JUDGE` provider and key.

Golden format (`ml/goldens/checklist/labels.jsonl`, one JSON object per
line):
    {"id": "...", "image_path": "...", "look_card": {...LookCard fields...},
     "labels": {"hair": true, "glasses": false, ...}}
`image_path` is relative to `ml/goldens/checklist/`. `labels` is the
human's ground truth per feature actually askable on that look card
(`features_for`'s output).

Run (from anywhere in the repo): `uv run python ml/evals/checklist.py`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from contracts import FeatureChecklistResult, LookCard  # noqa: E402
from ml.identity.checklist import score_checklist  # noqa: E402

GOLDENS_DIR = _REPO_ROOT / "ml" / "goldens" / "checklist"
REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass
class GoldenCase:
    id: str
    image_path: Path
    look_card: LookCard
    labels: dict[str, bool]  # ground truth, feature key -> present


@dataclass
class CaseScore:
    id: str
    mandatory_tp: int  # predicted present, labeled present
    mandatory_fp: int  # predicted present, labeled absent
    mandatory_total: int


@dataclass
class EvalReport:
    provider: str
    model: str
    cases: list[CaseScore] = field(default_factory=list)

    @property
    def mandatory_precision(self) -> float | None:
        """Of the mandatory-feature checks predicted `present`, what
        fraction were actually present per the human label. `None` when
        nothing was ever predicted present (undefined, not zero)."""
        tp = sum(c.mandatory_tp for c in self.cases)
        fp = sum(c.mandatory_fp for c in self.cases)
        if tp + fp == 0:
            return None
        return tp / (tp + fp)

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mandatory_precision": self.mandatory_precision,
            "n_cases": len(self.cases),
        }


def load_goldens(directory: Path = GOLDENS_DIR) -> list[GoldenCase]:
    labels_path = directory / "labels.jsonl"
    cases = []
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            GoldenCase(
                id=raw["id"],
                image_path=directory / raw["image_path"],
                look_card=LookCard(**raw["look_card"]),
                labels=raw["labels"],
            )
        )
    return cases


def score_case(golden: GoldenCase, result: FeatureChecklistResult) -> CaseScore:
    """Only mandatory checks count toward this metric — that's the actual
    gate master-candidate ranking uses; non-mandatory features are
    informative but not what precision measures here."""
    tp = fp = 0
    for check in result.checks:
        if not check.mandatory or not check.present:
            # Precision only counts rows the checklist predicted present
            # — a predicted-absent mandatory feature is a recall question,
            # not this metric's.
            continue
        label = golden.labels.get(check.feature)
        if label is None:
            continue  # no human ground truth for this feature on this case
        if label:
            tp += 1
        else:
            fp += 1
    return CaseScore(id=golden.id, mandatory_tp=tp, mandatory_fp=fp, mandatory_total=tp + fp)


async def run_eval(goldens: list[GoldenCase]) -> EvalReport:
    from app.llm.routing import resolve
    from ml.identity.faces import decode_jpeg_bgr, encode_crop_jpeg

    provider, model = resolve("judge")
    report = EvalReport(provider=provider.name, model=model)

    for golden in goldens:
        raw = golden.image_path.read_bytes()
        # Some master-batch renders on disk are genuinely PNG bytes under a
        # ".jpg" filename (fal.ai's output format wasn't pinned at download
        # time), and the vision call hardcodes an image/jpeg media type —
        # Anthropic's API 400s on the mismatch. cv2 detects the real format
        # from the byte header regardless of extension, so decode+re-encode
        # normalizes every golden to a real JPEG before scoring, matching
        # `master.py`'s own real candidate path.
        jpeg = encode_crop_jpeg(decode_jpeg_bgr(raw))
        result = await score_checklist(jpeg, golden.look_card, use_cache=False)
        report.cases.append(score_case(golden, result))

    return report


def main() -> None:
    goldens = load_goldens()
    report = asyncio.run(run_eval(goldens))

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"checklist_{report.provider}_{report.model.replace('/', '_')}.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    precision = report.mandatory_precision
    precision_str = f"{precision:.3f}" if precision is not None else "undefined (no positives)"
    print(f"provider={report.provider} model={report.model}")
    print(f"mandatory_precision={precision_str} (target >= 0.90)")
    print(f"n_cases={len(report.cases)}")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
