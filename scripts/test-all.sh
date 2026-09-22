#!/usr/bin/env bash
# Runs the full test suite. `make test` calls this, not `uv run pytest`
# directly — see the testpaths comment in pyproject.toml for why.
#
# Same split as scripts/mypy-all.sh, for the same reason: services/api,
# services/worker and services/scheduler each own a top-level `app/`
# package by design (CLAUDE.md §2), so a test file that imports `app.*`
# only resolves with that service's directory as cwd — and specifically
# via `python -m pytest` (which adds cwd to sys.path); the plain `pytest`
# console script does not. Contracts/storage are real installed packages,
# so they need neither trick and run once from the repo root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

status=0

echo "== repo root (contracts, storage, migrations) =="
uv run pytest tests/unit/contracts tests/unit/storage tests/integration packages || status=$?

echo "== services/worker (LLM provider layer, ...) =="
(cd services/worker && uv run python -m pytest ../../tests/unit/llm) || status=$?

# tests/unit/ml imports ml.evals.* (needs `ml` package, so `-m pytest` from
# the repo root — not services/worker as cwd) which in turn bootstraps
# services/worker onto sys.path itself for its own app.* imports (see
# ml/evals/extractor.py's docstring) — so this one needs no cwd change,
# just `-m` instead of the plain `pytest` console script.
echo "== tests/unit/ml (ml.evals.*, which bootstraps app.* itself) =="
uv run python -m pytest tests/unit/ml || status=$?

echo "== services/scheduler (cron entrypoints) =="
(cd services/scheduler && uv run python -m pytest ../../tests/unit/scheduler) || status=$?

echo "== services/api (routers, real Postgres + Playwright) =="
(cd services/api && uv run python -m pytest ../../tests/unit/api) || status=$?

echo "== tests/e2e (Gate 8 backend flow, needs services/api as cwd) =="
(cd services/api && uv run python -m pytest ../../tests/e2e) || status=$?

exit "$status"
