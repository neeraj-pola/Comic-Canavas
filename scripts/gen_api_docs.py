"""Task 8.13: regenerate `docs/openapi.json` + `docs/api.md` from the
real FastAPI app. Run from anywhere: `uv run python scripts/gen_api_docs.py`.

Delegates the actual `import app.main` to a `python -c` subprocess with
`cwd=services/api` — plain `python script.py` puts the SCRIPT's own
directory on `sys.path`, not `cwd` (only `-c`/`-m` do that), so this
can't just import `app.main` directly no matter what cwd it's launched
from; same constraint as this project's `scripts/test-all.sh`/
`mypy-all.sh` (CLAUDE.md §2 — each service owns its own top-level `app/`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_INLINE = "import json; from app.main import app; print(json.dumps(app.openapi()))"


def main() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "-c", _INLINE],
        cwd=REPO_ROOT / "services" / "api",
        check=True,
        capture_output=True,
        text=True,
    )
    schema = json.loads(result.stdout)
    (REPO_ROOT / "docs" / "openapi.json").write_text(json.dumps(schema, indent=2) + "\n")

    by_tag: dict[str, list[tuple[str, str, dict[str, object]]]] = {}
    for path, methods in schema["paths"].items():
        for method, op in methods.items():
            tag = (op.get("tags") or ["untagged"])[0]
            by_tag.setdefault(tag, []).append((method.upper(), path, op))

    lines = [
        "# API reference",
        "",
        "Generated from the real FastAPI app (`services/api/app/main.py`) via",
        "`app.openapi()` (task 8.13). Regenerate after any router change:",
        "`uv run python scripts/gen_api_docs.py` from the repo root.",
        "",
        "The raw schema is committed at `docs/openapi.json`; this file is a",
        "human-readable index into it, grouped by router.",
        "",
    ]
    for tag in sorted(by_tag):
        lines.append(f"## {tag}")
        lines.append("")
        for method, path, op in by_tag[tag]:
            summary = op.get("summary") or ""
            suffix = f" — {summary}" if summary else ""
            lines.append(f"- `{method} {path}`{suffix}")
        lines.append("")

    (REPO_ROOT / "docs" / "api.md").write_text("\n".join(lines) + "\n")
    print(f"wrote docs/openapi.json ({len(schema['paths'])} paths) and docs/api.md")


if __name__ == "__main__":
    main()
    sys.exit(0)
