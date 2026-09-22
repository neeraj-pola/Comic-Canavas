"""`model_json_schema()` snapshots for every contract.

A field changing shape without a matching snapshot update fails this test
— that's the point. To update a snapshot after a deliberate contract
change: `UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/contracts`, then look
at the diff (`git diff tests/unit/contracts/snapshots/`) before committing
it — the diff *is* the change under review.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from contracts import (
    Beat,
    BeatSheet,
    Candidate,
    CaptionPair,
    DayState,
    FeatureCheck,
    FeatureChecklistResult,
    ImagePair,
    ImagePrompt,
    LookCard,
    MasterCandidate,
    Panel,
    Script,
    SheetImage,
    Thumb,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

MODELS: dict[str, type[BaseModel]] = {
    "Beat": Beat,
    "BeatSheet": BeatSheet,
    "Panel": Panel,
    "Script": Script,
    "ImagePrompt": ImagePrompt,
    "Candidate": Candidate,
    "DayState": DayState,
    "ImagePair": ImagePair,
    "CaptionPair": CaptionPair,
    "Thumb": Thumb,
    "LookCard": LookCard,
    "FeatureCheck": FeatureCheck,
    "FeatureChecklistResult": FeatureChecklistResult,
    "MasterCandidate": MasterCandidate,
    "SheetImage": SheetImage,
}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_schema_matches_snapshot(name: str) -> None:
    schema = MODELS[name].model_json_schema()
    snapshot_path = SNAPSHOT_DIR / f"{name}.schema.json"
    rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"

    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(rendered)
        pytest.skip(f"UPDATE_SNAPSHOTS=1: wrote {snapshot_path}")

    if not snapshot_path.exists():
        pytest.fail(
            f"no snapshot for {name} at {snapshot_path}. "
            "Run `UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/contracts` to create it."
        )

    expected = snapshot_path.read_text()
    assert rendered == expected, (
        f"{name}.model_json_schema() no longer matches {snapshot_path.name}. "
        "If this field change is intentional, regenerate with "
        "`UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/contracts` and review the diff."
    )
