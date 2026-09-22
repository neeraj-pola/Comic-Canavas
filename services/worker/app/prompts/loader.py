"""Prompt loader.

Files: `<name>.v<N>.md` in this directory (`extractor.v1.md`, `script.v2.md`,
...), each a YAML front-matter block (`---`-delimited, at least `role` and
`temperature`; `notes` optional) followed by the prompt body.

`load_prompt(name)` picks the highest version present unless pinned via
`PROMPT_<NAME>=vN` (env). The exact "name.vN" string it picks
(`Prompt.versioned_name`) is what a node writes into `DayState.versions` —
e.g. `state.versions["extractor_prompt"] = prompt.versioned_name` — so every
run records exactly which prompt version produced it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_FILENAME_RE = re.compile(r"^(?P<name>[a-z0-9_]+)\.v(?P<version>\d+)\.md$")


@dataclass(frozen=True)
class Prompt:
    name: str  # "extractor"
    version: str  # "v3"
    role: str
    temperature: float
    notes: str | None
    text: str  # the prompt body, after front matter

    @property
    def versioned_name(self) -> str:
        return f"{self.name}.{self.version}"


class PromptNotFoundError(FileNotFoundError):
    def __init__(self, name: str, directory: Path) -> None:
        super().__init__(f"no prompt files named '{name}.v*.md' found in {directory}")


class PromptPinNotFoundError(FileNotFoundError):
    def __init__(self, name: str, version: str, directory: Path) -> None:
        env_var = f"PROMPT_{name.upper()}"
        super().__init__(f"{name}.{version}.md not found in {directory} (pinned via {env_var})")


class PromptFrontMatterError(ValueError):
    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")


def _parse(path: Path, *, name: str, version: str) -> Prompt:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise PromptFrontMatterError(path, "must start with '---' YAML front matter")

    _, _, rest = raw.partition("---")
    front_matter_raw, sep, body = rest.partition("---")
    if not sep:
        raise PromptFrontMatterError(path, "missing closing '---' for front matter")

    front_matter = yaml.safe_load(front_matter_raw) or {}
    if not isinstance(front_matter, dict):
        raise PromptFrontMatterError(path, "front matter must be a YAML mapping")
    if "role" not in front_matter:
        raise PromptFrontMatterError(path, "front matter must set 'role'")
    if "temperature" not in front_matter:
        raise PromptFrontMatterError(path, "front matter must set 'temperature'")

    return Prompt(
        name=name,
        version=version,
        role=str(front_matter["role"]),
        temperature=float(front_matter["temperature"]),
        notes=front_matter.get("notes"),
        text=body.strip("\n"),
    )


def _discover(directory: Path, name: str) -> dict[str, Path]:
    versions: dict[str, Path] = {}
    for path in directory.glob(f"{name}.v*.md"):
        match = _FILENAME_RE.match(path.name)
        if match and match.group("name") == name:
            versions[f"v{match.group('version')}"] = path
    return versions


def load_prompt(name: str, *, directory: Path | None = None) -> Prompt:
    search_dir = directory if directory is not None else Path(__file__).parent
    versions = _discover(search_dir, name)
    if not versions:
        raise PromptNotFoundError(name, search_dir)

    pin = os.environ.get(f"PROMPT_{name.upper()}")
    if pin:
        if pin not in versions:
            raise PromptPinNotFoundError(name, pin, search_dir)
        chosen = pin
    else:
        chosen = max(versions, key=lambda v: int(v[1:]))

    return _parse(versions[chosen], name=name, version=chosen)
