"""Identity eval: AUC >= 0.9 separating the approved character's own real
crops (positives) from the regularization set's different, generic comic
characters (negatives). Both are real, already-committed assets.
Bootstraps `services/worker` onto `sys.path` (same pattern as
`ml/identity/lookcard.py`) since `critic/identity.py` lives there.

Run (from anywhere in the repo): `uv run python ml/evals/critic_identity.py`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from app.critic.identity import score_identity  # noqa: E402
from contracts import LookCard  # noqa: E402

DATA_DIR = _REPO_ROOT / ".data"
REG_DIR = _REPO_ROOT / "ml" / "reg" / "comic_character"
REPORTS_DIR = Path(__file__).parent / "reports"
N_PER_CLASS = 20


def auc_score(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Mann-Whitney U formulation — the standard rank-based AUC, no
    sklearn dependency needed for one metric. Average ranks handle ties."""
    combined = sorted(positive_scores + negative_scores)
    ranks: dict[float, float] = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j] == combined[i]:
            j += 1
        average_rank = (i + 1 + j) / 2  # 1-indexed rank, averaged over the tied block
        for k in range(i, j):
            ranks[combined[k]] = average_rank
        i = j

    rank_sum = sum(ranks[s] for s in positive_scores)
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _positive_paths(n: int) -> list[Path]:
    sheet_dir = DATA_DIR / "people" / "me" / "sheet"
    paths = sorted(sheet_dir.glob("*.png"))
    return paths[:n]


def _negative_paths(n: int) -> list[Path]:
    paths = sorted(REG_DIR.glob("*.jpg"))
    step = max(1, len(paths) // n)
    return paths[::step][:n]


async def run_eval(n_per_class: int = N_PER_CLASS) -> dict[str, object]:
    identity_json = json.loads((DATA_DIR / "people" / "me" / "identity_model.json").read_text())
    look_card = LookCard(**identity_json["look_card"])

    positive_paths = _positive_paths(n_per_class)
    negative_paths = _negative_paths(n_per_class)

    positive_scores = []
    for path in positive_paths:
        score, _face_box = await score_identity(path.read_bytes(), look_card)
        positive_scores.append(score)

    negative_scores = []
    for path in negative_paths:
        score, _face_box = await score_identity(path.read_bytes(), look_card)
        negative_scores.append(score)

    auc = auc_score(positive_scores, negative_scores)
    return {
        "auc": auc,
        "target": 0.90,
        "n_positive": len(positive_scores),
        "n_negative": len(negative_scores),
        "mean_positive_score": sum(positive_scores) / len(positive_scores),
        "mean_negative_score": sum(negative_scores) / len(negative_scores),
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
    }


def main() -> None:
    report = asyncio.run(run_eval())
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "critic_identity_auc_2026-09-15.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"auc={report['auc']:.3f} (target >= 0.90)")
    print(f"mean_positive_score={report['mean_positive_score']:.3f}")
    print(f"mean_negative_score={report['mean_negative_score']:.3f}")
    print(f"n_positive={report['n_positive']} n_negative={report['n_negative']}")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
