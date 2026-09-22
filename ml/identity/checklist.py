"""VLM feature-checklist scorer — the replacement for ArcFace as the
identity gate: asks whether each of the look card's stable traits is
visible on one face crop, rather than whether the crop is geometrically
the same face as a reference photo. Reuses the existing `judge` LLM role.

Bootstraps `services/worker` onto `sys.path`, same pattern as
`ml/identity/lookcard.py`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from pydantic import BaseModel  # noqa: E402

from app.llm.base import ContentPart, Message  # noqa: E402
from app.llm.routing import resolve  # noqa: E402
from app.prompts.loader import load_prompt  # noqa: E402
from contracts import FeatureCheck, FeatureChecklistResult, LookCard, LookCardField  # noqa: E402

# Same set nodes/prompts.py's _describe_look_card and lora_dataset.py's
# _caption_prefix already treat as "nothing to say here" — a field with
# one of these values can't be answered yes/no (a person with no glasses
# can't "pass" a glasses check), so it's skipped rather than asked.
_NONE_VALUES = {"", "none", "none noted", "not enough information"}


class _VisionFeatureAnswer(BaseModel):
    feature: str  # echoed key; reconciled against the asked-for list in code
    present: bool
    note: str = ""


class _VisionChecklistAnswers(BaseModel):
    """The vision call's actual structured-output schema. `artifacts_clean`
    rides along on a call already being paid for, an OCR/artifact
    rejection signal without a new OCR dependency."""

    answers: list[_VisionFeatureAnswer]
    artifacts_clean: bool
    artifact_note: str = ""


@dataclass(frozen=True)
class ChecklistFeature:
    key: LookCardField
    expected: str
    mandatory: bool


_cache: dict[str, FeatureChecklistResult] = {}


def clear_cache() -> None:
    _cache.clear()


def features_for(look_card: LookCard) -> list[ChecklistFeature]:
    """One feature per look-card line that actually says something. A
    `mandatory` field whose value is unanswerable ("none"/"not enough
    information") is dropped from the asked-for list entirely, not kept
    as an unpassable mandatory gate — a person with no facial hair can't
    be failed forever for lacking facial hair."""
    fields: list[LookCardField] = [
        "hair",
        "glasses",
        "skin_tone",
        "face_shape",
        "signature_outfit",
        "distinguishing",
    ]
    features = []
    for field in fields:
        value = getattr(look_card, field).strip()
        if value.lower() in _NONE_VALUES:
            continue
        features.append(
            ChecklistFeature(key=field, expected=value, mandatory=field in look_card.mandatory)
        )
    return features


def _cache_key(jpeg: bytes, features: list[ChecklistFeature], model: str) -> str:
    image_hash = hashlib.sha256(jpeg).hexdigest()
    features_hash = hashlib.sha256(
        json.dumps([(f.key, f.expected) for f in features], sort_keys=True).encode()
    ).hexdigest()
    return f"{image_hash}:{features_hash}:{model}"


def _find_answer(key: str, answers: list[_VisionFeatureAnswer]) -> _VisionFeatureAnswer | None:
    """Exact match first; falls back to a key-prefix match ("hair" also
    matches a returned `feature` of "hair: short black hair") — defense in
    depth against the model echoing a whole line instead of just the key,
    on top of the prompt itself asking for an exact echo."""
    for answer in answers:
        if answer.feature.strip() == key:
            return answer
    for answer in answers:
        if answer.feature.strip().lower().startswith(key.lower()):
            return answer
    return None


def _reconcile(
    features: list[ChecklistFeature], answers: list[_VisionFeatureAnswer]
) -> list[FeatureCheck]:
    checks = []
    for feature in features:
        answer = _find_answer(feature.key, answers)
        if answer is None:
            checks.append(
                FeatureCheck(
                    feature=feature.key,
                    expected=feature.expected,
                    present=False,
                    mandatory=feature.mandatory,
                    note="no answer",
                )
            )
        else:
            checks.append(
                FeatureCheck(
                    feature=feature.key,
                    expected=feature.expected,
                    present=answer.present,
                    mandatory=feature.mandatory,
                    note=answer.note,
                )
            )
    return checks


def _build_messages(
    image_jpeg: bytes, features: list[ChecklistFeature], prompt_text: str
) -> list[Message]:
    feature_list = "\n".join(f"- {f.key}: {f.expected}" for f in features)
    content: list[ContentPart] = [
        {"type": "text", "text": f"Features to check:\n{feature_list}"},
        {"type": "image", "b64": base64.b64encode(image_jpeg).decode("ascii")},
    ]
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": content},
    ]


async def score_checklist(
    face_crop_jpeg: bytes, look_card: LookCard, *, use_cache: bool = True
) -> FeatureChecklistResult:
    features = features_for(look_card)
    prompt = load_prompt("checklist")
    provider, model = resolve("judge")
    model_id = f"{provider.name}:{model}"

    image_sha256 = hashlib.sha256(face_crop_jpeg).hexdigest()
    key = _cache_key(face_crop_jpeg, features, model_id)
    if use_cache and key in _cache:
        return _cache[key]

    messages = _build_messages(face_crop_jpeg, features, prompt.text)
    vision_answers, _result = await provider.structured(
        messages, _VisionChecklistAnswers, model=model, temperature=prompt.temperature
    )

    checks = _reconcile(features, vision_answers.answers)
    score = sum(1 for c in checks if c.present) / len(checks) if checks else 1.0
    mandatory_passed = all(c.present for c in checks if c.mandatory)

    result = FeatureChecklistResult(
        checks=checks,
        score=score,
        mandatory_passed=mandatory_passed,
        artifacts_clean=vision_answers.artifacts_clean,
        artifact_note=vision_answers.artifact_note,
        image_sha256=image_sha256,
        model=model_id,
    )
    if use_cache:
        _cache[key] = result
    return result
