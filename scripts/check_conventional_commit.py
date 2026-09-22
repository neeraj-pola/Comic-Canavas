#!/usr/bin/env python3
"""Reject commit messages that aren't Conventional Commits (task 0.4).

CLAUDE.md §0.1: "Small commits, conventional messages. `feat(extractor): …`,
`fix(critic): …`, `test(api): …`, `docs(adr): …`." Wired as the `commit-msg`
stage in `.pre-commit-config.yaml`. `golden` is CLAUDE.md's own addition to
the standard type list, not a standard Conventional Commits type — required
for any change under `ml/goldens/` ("change only via a PR titled `golden:`
with a reason", same section).
"""

from __future__ import annotations

import re
import sys

TYPES = [
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
    "golden",
]

# type(optional-scope)!?: subject — e.g. "feat(extractor): add v2 prompt"
PATTERN = re.compile(rf"^({'|'.join(TYPES)})(\([\w./-]+\))?!?: .+")


def _is_generated(message: str) -> bool:
    """git/GitHub write these themselves; nobody hand-types them."""
    prefixes = ("Merge ", 'Revert "', "fixup!")
    return message.startswith(prefixes)


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "check_conventional_commit: expected the commit-msg file path as argv[1]",
            file=sys.stderr,
        )
        return 2

    with open(argv[0], encoding="utf-8") as f:
        message = f.readline().rstrip("\n")

    if _is_generated(message):
        return 0

    if PATTERN.match(message):
        return 0

    print(
        "✗ Commit message rejected — not a Conventional Commit (CLAUDE.md §0.1).\n"
        f"  Got:      {message!r}\n"
        f"  Expected: <type>(<scope>): <subject>, e.g. 'feat(extractor): add v2 prompt'\n"
        f"  Types:    {', '.join(TYPES)}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
