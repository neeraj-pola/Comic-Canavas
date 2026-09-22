"""Alignment eval: the correct panel should score higher than a shuffled
caption at least 90% of the time. Uses 20 real, already-generated panels
with known real action captions rather than synthetic goldens — real
images, real intended content, zero new generation cost (SigLIP inference
runs locally on CPU).

Run (from anywhere in the repo): `uv run python ml/evals/critic_alignment.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from app.critic.alignment import score_alignment  # noqa: E402
from ml.identity.faces import decode_jpeg_bgr  # noqa: E402

CANDIDATES_DIR = _REPO_ROOT / ".data" / "candidates"
REPORTS_DIR = Path(__file__).parent / "reports"

# The real action captions for each of the 20 real, already-generated panels.
PANEL_ACTIONS: dict[int, str] = {
    1: "typing on a laptop at a desk",
    2: "pouring coffee in a kitchen",
    3: "walking down a city street",
    4: "lifting a dumbbell mid-rep at the gym",
    5: "reading a book by lamp light in bed",
    6: "waiting under an umbrella in the rain",
    7: "talking with a friend over coffee at a cafe",
    8: "washing dishes at a kitchen sink",
    9: "on a video call at a desk",
    10: "cooking dinner at the stove",
    11: "jogging on a park path",
    12: "stretching on a mat at the gym",
    13: "waking up and stretching in bed",
    14: "reading on a phone during a train commute",
    15: "ordering coffee at a cafe counter",
    16: "doing laundry",
    17: "writing notes in a notebook at a desk",
    18: "washing vegetables at a kitchen sink",
    19: "sitting on a park bench outdoors",
    20: "high-fiving a friend at the gym",
}


def _first_image_path(panel_id: int) -> Path:
    panel_dir = CANDIDATES_DIR / f"panel-{panel_id}"
    return sorted(panel_dir.glob("*.png"))[0]


def run_eval() -> dict[str, object]:
    panel_ids = sorted(PANEL_ACTIONS)
    n = len(panel_ids)

    wins = 0
    rows = []
    for i, panel_id in enumerate(panel_ids):
        image_bgr = decode_jpeg_bgr(_first_image_path(panel_id).read_bytes())
        correct_caption = PANEL_ACTIONS[panel_id]
        # Deterministic "shuffle": pair with the next panel's caption
        # (wrapping around) — guaranteed different scene, no randomness
        # to make results non-reproducible.
        shuffled_panel_id = panel_ids[(i + 1) % n]
        shuffled_caption = PANEL_ACTIONS[shuffled_panel_id]

        correct_score = score_alignment(image_bgr, correct_caption)
        shuffled_score = score_alignment(image_bgr, shuffled_caption)
        win = correct_score > shuffled_score
        wins += int(win)
        rows.append(
            {
                "panel_id": panel_id,
                "correct_caption": correct_caption,
                "correct_score": correct_score,
                "shuffled_caption": shuffled_caption,
                "shuffled_score": shuffled_score,
                "win": win,
            }
        )

    win_rate = wins / n
    return {"win_rate": win_rate, "target": 0.90, "n_panels": n, "rows": rows}


def main() -> None:
    report = run_eval()
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "critic_alignment_winrate_2026-09-15.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"win_rate={report['win_rate']:.3f} (target >= 0.90)")
    print(f"n_panels={report['n_panels']}")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
