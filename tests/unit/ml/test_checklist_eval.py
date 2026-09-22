"""The pure, network-free half of ml/evals/checklist.py — `load_goldens`
and `score_case`. `run_eval`/`main` need a real LLM_JUDGE provider and
key; not exercised by `make test`. These tests build a synthetic
labels.jsonl under `tmp_path` rather than relying on a committed golden
set.
"""

from __future__ import annotations

import json
from pathlib import Path

from contracts import FeatureCheck, FeatureChecklistResult
from ml.evals.checklist import GoldenCase, load_goldens, score_case


def _write_labels(directory: Path, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def _look_card_dict(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "hair": "short black wavy hair",
        "glasses": "black rectangular glasses",
        "skin_tone": "medium",
        "face_shape": "oval",
        "signature_outfit": "not enough information",
        "distinguishing": "a small mole above the left eyebrow",
        "gender_term": "man",
        "mandatory": ["glasses", "hair"],
    }
    fields.update(overrides)
    return fields


def _check(feature: str, present: bool, *, mandatory: bool) -> FeatureCheck:
    return FeatureCheck(feature=feature, expected="x", present=present, mandatory=mandatory)


def test_load_goldens_reads_jsonl_rows(tmp_path: Path) -> None:
    (tmp_path / "crop.jpg").write_bytes(b"fake-jpeg")
    _write_labels(
        tmp_path,
        [
            {
                "id": "case-1",
                "image_path": "crop.jpg",
                "look_card": _look_card_dict(),
                "labels": {"hair": True, "glasses": False},
            }
        ],
    )
    goldens = load_goldens(tmp_path)
    assert len(goldens) == 1
    assert isinstance(goldens[0], GoldenCase)
    assert goldens[0].image_path == tmp_path / "crop.jpg"
    assert goldens[0].look_card.hair == "short black wavy hair"


def test_load_goldens_skips_blank_lines(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "labels.jsonl").write_text(
        json.dumps(
            {
                "id": "case-1",
                "image_path": "crop.jpg",
                "look_card": _look_card_dict(),
                "labels": {},
            }
        )
        + "\n\n"
    )
    assert len(load_goldens(tmp_path)) == 1


def test_score_case_counts_true_positives_and_false_positives() -> None:
    golden = GoldenCase(
        id="case-1",
        image_path=Path("crop.jpg"),
        look_card=None,  # type: ignore[arg-type]  # not read by score_case
        labels={"hair": True, "glasses": False},
    )
    result = FeatureChecklistResult(
        checks=[
            _check("hair", present=True, mandatory=True),  # TP: labeled true, predicted true
            _check("glasses", present=True, mandatory=True),  # FP: labeled false, predicted true
        ],
        score=1.0,
        mandatory_passed=True,
        artifacts_clean=True,
        image_sha256="x",
        model="mock:mock",
    )
    score = score_case(golden, result)
    assert score.mandatory_tp == 1
    assert score.mandatory_fp == 1
    assert score.mandatory_total == 2


def test_score_case_ignores_non_mandatory_checks() -> None:
    golden = GoldenCase(
        id="case-1",
        image_path=Path("crop.jpg"),
        look_card=None,  # type: ignore[arg-type]  # not read by score_case
        labels={"skin_tone": False},
    )
    result = FeatureChecklistResult(
        checks=[_check("skin_tone", present=True, mandatory=False)],
        score=1.0,
        mandatory_passed=True,
        artifacts_clean=True,
        image_sha256="x",
        model="mock:mock",
    )
    score = score_case(golden, result)
    assert score.mandatory_total == 0


def test_score_case_ignores_predicted_absent_checks() -> None:
    """A predicted-absent mandatory feature isn't a false positive by this
    metric's definition (precision: of the ones the checklist says are
    present, how many really are) — predicted-absent cases belong to
    recall, a different question."""
    golden = GoldenCase(
        id="case-1",
        image_path=Path("crop.jpg"),
        look_card=None,  # type: ignore[arg-type]  # not read by score_case
        labels={"hair": True},
    )
    result = FeatureChecklistResult(
        checks=[_check("hair", present=False, mandatory=True)],
        score=0.0,
        mandatory_passed=False,
        artifacts_clean=True,
        image_sha256="x",
        model="mock:mock",
    )
    score = score_case(golden, result)
    assert score.mandatory_tp == 0
    assert score.mandatory_fp == 0
    assert score.mandatory_total == 0
