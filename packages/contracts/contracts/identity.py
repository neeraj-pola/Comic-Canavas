"""The look card — a module-boundary contract between identity onboarding
(`ml/identity/`) and the prompt writer (`services/worker/app/nodes/prompts.py`).

Stable, nameable traits a person would recognize about themselves — not an
exhaustive physical description (identity itself is carried by the image
generator's own reference mechanism, not by prose). The prompt writer uses
these to build `ImagePrompt.character_clause` deterministically, not by
asking an LLM to restate them from memory each panel — the point of a look
card is that these facts show up in every panel's prompt the same way every
time, not "usually."

Identity is a designed comic character (a "master" design, approved once),
not a photo pushed through stylization — every identity mechanism tried
(a photo-trained LoRA, Character Reference, PuLID-Flux) lost exact facial
geometry under real comic stylization, and only coarse traits (age, gender,
hair, glasses) survived. The look card is the ground truth a VLM feature
checklist checks master candidates against — hence `mandatory`/`edits`
below, and `FeatureCheck`/`FeatureChecklistResult`/`MasterCandidate`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LookCardField = Literal[
    "hair", "glasses", "skin_tone", "face_shape", "signature_outfit", "distinguishing"
]

_DEFAULT_MANDATORY: list[LookCardField] = ["glasses", "hair", "distinguishing"]


class LookCard(BaseModel):
    hair: str
    glasses: str
    skin_tone: str
    face_shape: str
    signature_outfit: str
    distinguishing: str
    # Deliberately NOT vision-inferred — guessing gender from photos risks
    # misgendering and isn't more reliable than just asking the person.
    # Empty string means "not set"; callers building a character_clause
    # or LoRA training caption fall back to a neutral "a person".
    gender_term: str = ""
    # Same reasoning as gender_term: test renders skewed younger than the
    # reference photos without an age anchor in the prompt — a short phrase
    # like "24-year-old", not vision-guessed.
    age_descriptor: str = ""
    # Which look-card lines the feature checklist treats as pass/fail gates
    # rather than nice-to-haves, named by the actual field each aspect lives
    # on (frame shape -> glasses, facial hair -> distinguishing, hair
    # direction -> hair).
    mandatory: list[LookCardField] = Field(default_factory=lambda: list(_DEFAULT_MANDATORY))
    # Free-text audit trail of chip taps from the master-design loop
    # ("thicker glasses", ...). Each chip rewrites exactly one field above;
    # this records that it was asked for, so a later regeneration can tell
    # a person-corrected trait apart from a vision-inferred one.
    edits: list[str] = Field(default_factory=list)


class FeatureCheck(BaseModel):
    """One look-card line, judged present or not on one face crop."""

    feature: LookCardField
    expected: str  # the look-card line verbatim, so a stored result stays readable later
    present: bool
    mandatory: bool  # copied from LookCard.mandatory at scoring time
    note: str = ""  # one short phrase of evidence — the only free text the VLM adds


class FeatureChecklistResult(BaseModel):
    """The identity metric: "is every feature of the approved character
    present" rather than "is this geometrically the same face" — a VLM
    checklist result, crossing from master ranking into the critic.
    """

    checks: list[FeatureCheck]
    score: float  # passed / total, computed in code, never asked of the VLM
    mandatory_passed: bool  # every mandatory check present
    artifacts_clean: bool
    artifact_note: str = ""
    image_sha256: str  # cache key + provenance for the labeled eval set
    model: str  # e.g. "anthropic:claude-sonnet-4-6" — which judge produced this result


class MasterCandidate(BaseModel):
    """One ranked stylized master-design candidate. Crosses into the
    onboarding "which one feels like you" screen, so it's a contract, not a
    local dataclass. `prompt`/`seed` are kept so a later character sheet can
    regenerate from the exact phrasing the approved master came from."""

    id: str  # f"{seed}-{index}"
    url: str  # storage URL under people/{id}/master_candidates/
    seed: int
    source_crop_index: int  # which of the best front crops drove the canny structure
    prompt: str
    checklist: FeatureChecklistResult
    style_score: float  # DINOv2 cosine to the regularization set
    arcface_agreement: float | None  # weak signal only; None if no face
    rank_score: float  # the combined number actually used to sort candidates


class SheetImage(BaseModel):
    """One curated character-sheet image — a reference-preserving variation
    generated from the approved master (not the raw photos). Crosses into
    the onboarding "meet your character" screen and is the character LoRA's
    training set, so it's a contract, not a local dataclass."""

    id: str  # f"{view_index}-{expression_index}"
    url: str  # storage URL under people/{id}/sheet/
    view: str
    expression: str
    checklist: FeatureChecklistResult
    dinov2_to_master: float  # face-crop DINOv2 cosine to the master (the curation gate)
    kept: bool  # mandatory_passed and dinov2_to_master >= 0.7


__all__ = [
    "FeatureCheck",
    "FeatureChecklistResult",
    "LookCard",
    "LookCardField",
    "MasterCandidate",
    "SheetImage",
]
