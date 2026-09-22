"""Style eval: on/off style separation with accuracy >= 0.9. Uses two
real, already-on-disk assets at zero generation cost: the regularization
set (150 real in-style comic renders) as positives, and the person's raw
onboarding photos (real photographs — definitively off-style) as
negatives.

Run (from anywhere in the repo): `uv run python ml/evals/critic_style.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

from ml.identity.faces import decode_jpeg_bgr, load_bgr  # noqa: E402
from ml.identity.style_score import load_reference_bank, style_score  # noqa: E402

DATA_DIR = _REPO_ROOT / ".data"
REG_DIR = _REPO_ROOT / "ml" / "reg" / "comic_character"
RAW_DIR = DATA_DIR / "people" / "me" / "raw"
REPORTS_DIR = Path(__file__).parent / "reports"


def best_threshold_accuracy(
    positive_scores: list[float], negative_scores: list[float]
) -> tuple[float, float]:
    """Sweeps every score as a candidate threshold (classify >= threshold
    as "on-style") and returns the `(best_accuracy, threshold)` pair —
    the standard way to report "does this metric separate the two
    classes" without assuming one specific cutoff in advance."""
    candidates = sorted(set(positive_scores + negative_scores))
    best_accuracy = 0.0
    best_threshold = candidates[0]
    n = len(positive_scores) + len(negative_scores)
    for threshold in candidates:
        correct = sum(1 for s in positive_scores if s >= threshold)
        correct += sum(1 for s in negative_scores if s < threshold)
        accuracy = correct / n
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_accuracy, best_threshold


def run_eval(n_per_class: int = 20) -> dict[str, object]:
    bank = load_reference_bank()

    positive_paths = sorted(REG_DIR.glob("*.jpg"))[:n_per_class]
    positive_scores = [style_score(decode_jpeg_bgr(p.read_bytes()), bank) for p in positive_paths]

    raw_paths = sorted(
        p for p in RAW_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic")
    )[:n_per_class]
    negative_scores = [style_score(load_bgr(p), bank) for p in raw_paths]

    accuracy, threshold = best_threshold_accuracy(positive_scores, negative_scores)
    return {
        "accuracy": accuracy,
        "threshold": threshold,
        "target": 0.90,
        "n_positive": len(positive_scores),
        "n_negative": len(negative_scores),
        "mean_positive_score": sum(positive_scores) / len(positive_scores),
        "mean_negative_score": sum(negative_scores) / len(negative_scores),
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
    }


def main() -> None:
    report = run_eval()
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "critic_style_accuracy_2026-09-15.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"accuracy={report['accuracy']:.3f} (target >= 0.90), threshold={report['threshold']:.3f}"
    )
    print(f"mean_positive_score={report['mean_positive_score']:.3f}")
    print(f"mean_negative_score={report['mean_negative_score']:.3f}")
    print(f"n_positive={report['n_positive']} n_negative={report['n_negative']}")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
