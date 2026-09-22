"""The real Bradley-Terry reward head, trained on real preference pairs
(`ml/goldens/pairs/heldout_week.jsonl`) instead of the hand-set linear
combo (`services/worker/app/critic/reward_head.py`).

No Modal/GPU here: a Bradley-Terry head over 4 already-computed scalar
features (identity/style/alignment/detail, already stored in
`candidates.scores`) is a single `nn.Linear(4, 1, bias=False)` — logistic
regression on a 4-dimensional feature difference. Training this on a
real week's pairs takes milliseconds on CPU; provisioning a GPU for it
would be pure overhead. GPU compute is genuinely needed for fine-tuning
actual language/diffusion models, not this one.

On mock-provider data specifically, `identity` and `detail` carry almost
no learnable signal in a difference formulation (no real face or art to
score), so only `style`/`alignment` meaningfully differ between
candidates — whatever the learned head discovers on that data is a
genuine but modest result; a week of genuinely varied real candidates is
what this mechanism is actually built for. The eval gate decides whether
a result any given dataset produces is good enough to ship.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import torch
from dotenv import find_dotenv, load_dotenv
from psycopg.types.json import Json
from torch import nn

FEATURES = ("identity", "style", "alignment", "detail")

# Mirrors services/worker/app/critic/reward_head.py's real WEIGHTS —
# duplicated (not imported) because that module lives inside services/
# worker's own `app` package (a separate per-service namespace) and this
# module must stay import-safe from anywhere.
HAND_SET_WEIGHTS: dict[str, float] = {
    "identity": 0.35,
    "style": 0.25,
    "alignment": 0.25,
    "detail": 0.15,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
HELDOUT_PATH = REPO_ROOT / "ml" / "goldens" / "pairs" / "heldout_week.jsonl"
REPORTS_DIR = REPO_ROOT / "ml" / "evals" / "reports"
CHECKPOINTS_DIR = REPO_ROOT / "ml" / "reward_model" / "checkpoints"


@dataclass(frozen=True)
class PreferencePair:
    chosen: np.ndarray
    rejected: np.ndarray


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _feature_vector(scores: dict[str, float]) -> np.ndarray:
    return np.array([_clip01(scores.get(f, 0.0)) for f in FEATURES], dtype=np.float32)


def load_pairs(path: Path = HELDOUT_PATH) -> list[PreferencePair]:
    pairs = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            pairs.append(
                PreferencePair(
                    chosen=_feature_vector(row["chosen_scores"]),
                    rejected=_feature_vector(row["rejected_scores"]),
                )
            )
    return pairs


def split_train_eval(
    pairs: list[PreferencePair], *, eval_fraction: float = 0.25, seed: int = 42
) -> tuple[list[PreferencePair], list[PreferencePair]]:
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, round(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]


class RewardHead(nn.Module):
    """One linear layer, no bias — a direct, literal Bradley-Terry utility
    function over the four critic features, matching the hand-set head's
    own shape for a fair comparison."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(len(FEATURES), 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def weights(self) -> dict[str, float]:
        raw = self.linear.weight.detach().numpy().flatten()
        return dict(zip(FEATURES, (float(w) for w in raw), strict=True))


def train_reward_head(
    pairs: list[PreferencePair], *, epochs: int = 500, lr: float = 0.05
) -> RewardHead:
    model = RewardHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    chosen = torch.tensor(np.stack([p.chosen for p in pairs]))
    rejected = torch.tensor(np.stack([p.rejected for p in pairs]))
    target = torch.ones(len(pairs))  # "chosen" always wins by definition of the data

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(chosen) - model(rejected)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()
    return model


def pairwise_accuracy(weights: dict[str, float], pairs: list[PreferencePair]) -> float:
    """Fraction of pairs where `weights . chosen > weights . rejected` —
    i.e. this weighting would have picked the same candidate the person
    (or the dataset's own construction) preferred."""
    w = np.array([weights[f] for f in FEATURES])
    correct = sum(1 for p in pairs if float(w @ p.chosen) > float(w @ p.rejected))
    return correct / len(pairs)


def export_onnx(model: RewardHead, path: Path) -> None:
    """Torch's default ONNX exporter is the newer dynamo-based one, which
    needs `onnxscript` (not installed — this project depends on
    `onnxruntime`, the separate inference package, not the export-side
    one). `dynamo=False` uses the older TorchScript-based exporter, which
    needs nothing extra and is more than adequate for a single
    `nn.Linear`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, len(FEATURES))
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["features"],
        output_names=["reward"],
        dynamic_axes={"features": {0: "batch"}, "reward": {0: "batch"}},
        dynamo=False,
    )


def _insert_checkpoint_and_run(
    *, metrics: dict[str, Any], onnx_path: Path, snapshot_hash: str
) -> str:
    load_dotenv(find_dotenv(usecwd=True))
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return ""
    checkpoint_id = f"reward_head_v1_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    with psycopg.connect(database_url) as conn:
        schema = os.environ.get("DB_SCHEMA")
        if schema:
            conn.execute(f'SET search_path TO "{schema}", public')
        conn.execute(
            "INSERT INTO checkpoints (id, kind, data_snapshot, metrics, artifact_url) "
            "VALUES (%s, 'reward_model', %s, %s, %s)",
            (checkpoint_id, snapshot_hash, Json(metrics), str(onnx_path)),
        )
        conn.execute(
            "INSERT INTO training_runs (kind, config, snapshot_hash, metrics) "
            "VALUES ('reward_model', %s, %s, %s)",
            (Json({"features": list(FEATURES)}), snapshot_hash, Json(metrics)),
        )
        conn.commit()
    return checkpoint_id


def main() -> None:
    pairs = load_pairs()
    train_pairs, eval_pairs = split_train_eval(pairs)

    model = train_reward_head(train_pairs)
    learned_weights = model.weights()

    learned_acc = pairwise_accuracy(learned_weights, eval_pairs)
    hand_set_acc = pairwise_accuracy(HAND_SET_WEIGHTS, eval_pairs)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = CHECKPOINTS_DIR / "reward_head_v1.onnx"
    export_onnx(model, onnx_path)

    metrics = {
        "n_pairs": len(pairs),
        "n_train": len(train_pairs),
        "n_eval": len(eval_pairs),
        "learned_weights": learned_weights,
        "hand_set_weights": HAND_SET_WEIGHTS,
        "learned_pairwise_accuracy": learned_acc,
        "hand_set_pairwise_accuracy": hand_set_acc,
        "beats_champion": learned_acc >= hand_set_acc + 0.01,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"reward_model_{datetime.now(UTC).strftime('%Y-%m-%d')}.json"
    report_path.write_text(json.dumps(metrics, indent=2))

    snapshot_hash = f"heldout_week:{len(pairs)}pairs"
    checkpoint_id = _insert_checkpoint_and_run(
        metrics=metrics, onnx_path=onnx_path, snapshot_hash=snapshot_hash
    )

    from ml.experiment_tracking import log_run

    run_url = log_run(
        kind="reward_model",
        config={"features": list(FEATURES), "epochs": 500, "lr": 0.05},
        snapshot_hash=snapshot_hash,
        metrics=metrics,
    )

    print(json.dumps(metrics, indent=2))
    print(f"checkpoint_id={checkpoint_id or '(no DATABASE_URL, not recorded)'}")
    print(f"report={report_path}")
    print(f"onnx={onnx_path}")
    print(f"experiment_run={run_url}")


if __name__ == "__main__":
    main()
