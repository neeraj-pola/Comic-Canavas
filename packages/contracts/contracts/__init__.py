"""Pydantic contracts shared by the API and worker.

`model_json_schema()` for every model here is snapshot-tested
(`tests/unit/contracts/test_schema_snapshots.py`) — changing a field without
updating `tests/unit/contracts/snapshots/*.json` fails `make test`.
"""

from .day import Beat, BeatSheet, Candidate, DayState, ImagePrompt, Panel, Script
from .feedback import CaptionPair, ImagePair, Thumb
from .identity import (
    FeatureCheck,
    FeatureChecklistResult,
    LookCard,
    LookCardField,
    MasterCandidate,
    SheetImage,
)

__all__ = [
    "Beat",
    "BeatSheet",
    "Candidate",
    "CaptionPair",
    "DayState",
    "FeatureCheck",
    "FeatureChecklistResult",
    "ImagePair",
    "ImagePrompt",
    "LookCard",
    "LookCardField",
    "MasterCandidate",
    "Panel",
    "Script",
    "SheetImage",
    "Thumb",
]
