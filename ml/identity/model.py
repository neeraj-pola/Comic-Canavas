"""The identity model record — one row per person, filled in incrementally:
the averaged ArcFace embedding, the look card, Leonardo Character
Reference ids, and the Flux LoRA's URL + trigger token all live on the
same record, matching `images/base.py`'s `IdentityRef` shape — this is
what the real `identity_models` table looks like, one row per person.

`embedding` is a weak ranking signal, not the identity gate; `leonardo_ref_
ids`/`flux_lora_url` are superseded for generation once a master design
exists. The `master_*`/`style_card_version`/`chip_rounds` fields are the
source of truth for identity.

Storage is an interim seam ahead of the real Postgres table (same pattern
as `InMemoryMemoryStore`/`InMemoryCostEventSink` in services/worker):
`JsonIdentityModelStore` persists to a local JSON file per person for now;
swap for a Postgres-backed store later without touching callers.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from contracts import LookCard


class IdentityModel(BaseModel):
    person_id: str

    # A weak ranking/tie-break signal, not the identity metric.
    embedding: list[float] | None = None  # averaged, L2-normalized ArcFace embedding (512-d)
    source_photo_count: int | None = None
    mean_self_cosine: float | None = None

    look_card: LookCard | None = None

    # Superseded for generation once a master exists; kept for the
    # candidate-scoring step and as a pre-master fallback.
    leonardo_ref_ids: list[str] = Field(default_factory=list)

    # Baseline experiment, not the production path.
    flux_lora_url: str | None = None
    trigger_token: str | None = None

    # The character-first master design: the approved comic character
    # design itself, not a photo — this is what every future panel and a
    # LoRA retrain actually references. Not storing the full candidate
    # history here on purpose: candidate renders live as files under
    # people/{id}/master_candidates/, and master_candidate_id + master_seed
    # are enough to find the winner again.
    master_path: str | None = None  # storage key, people/{id}/master.png
    master_approved_at: datetime | None = None
    master_candidate_id: str | None = None  # which of the top candidates the person picked
    master_seed: int | None = None  # reproducibility for the character sheet
    style_card_version: str | None = None  # the style card the master was designed under
    chip_rounds: int = 0  # how many chip-edit rounds it took to approve

    # Character sheet: curated reference-preserving variations generated
    # from the master (not the raw photos) — the character LoRA's training
    # set and the onboarding "meet your character" previews. Paths only
    # (storage keys under people/{id}/sheet/); per-image scoring detail
    # lives in the eval report.
    sheet_paths: list[str] = Field(default_factory=list)
    sheet_generated_at: datetime | None = None

    # Which generation path this person actually uses, decided by
    # comparing a trained character LoRA against the reference path on the
    # same evidence — the loser stays documented, not deleted (a future
    # retrain could still win the gate later). `lora_checkpoint` records
    # which trained checkpoint was evaluated, independent of whether it won.
    generation_path: Literal["reference", "lora"] | None = None
    lora_checkpoint: str | None = None


class IdentityModelStore(Protocol):
    def save(self, model: IdentityModel) -> None: ...
    def load(self, person_id: str) -> IdentityModel | None: ...


class JsonIdentityModelStore:
    """Interim store: one JSON file per person under `root`, matching
    `Storage`'s real `people/{id}/...` key convention — swap for a real
    Postgres `identity_models` table without touching callers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, person_id: str) -> Path:
        return self.root / person_id / "identity_model.json"

    def save(self, model: IdentityModel) -> None:
        path = self._path(model.person_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model.model_dump_json(indent=2))

    def load(self, person_id: str) -> IdentityModel | None:
        path = self._path(person_id)
        if not path.exists():
            return None
        return IdentityModel.model_validate_json(path.read_text())

    def get_or_create(self, person_id: str) -> IdentityModel:
        return self.load(person_id) or IdentityModel(person_id=person_id)
