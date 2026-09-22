"""Style card loader — `style_card.yaml`'s line/palette/texture/camera/
character/environment/negative_standard, typed so a malformed edit to that
file fails loudly here rather than silently dropping a field into a panel
prompt. Loaded by the prompt writer (`nodes/prompts.py`).

`version` is recorded on `IdentityModel.style_card_version` so a master
design can be traced back to the exact visual language it was designed
under; `style_phrase` is the one canonical style string every render draws
from verbatim, so the look never gets reworded by hand in multiple places.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

_DEFAULT_PATH = Path(__file__).parent / "style_card.yaml"


class CharacterStyle(BaseModel):
    rules: list[str]
    avoid: list[str]


class EnvironmentStyle(BaseModel):
    rules: list[str]


class StyleCard(BaseModel):
    version: str
    style_phrase: str
    line: dict[str, str]
    palette: dict[str, str]
    texture: dict[str, str]
    camera: dict[str, str]
    character: CharacterStyle
    environment: EnvironmentStyle
    negative_standard: list[str]


def load_style_card(path: Path = _DEFAULT_PATH) -> StyleCard:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return StyleCard.model_validate(raw)
