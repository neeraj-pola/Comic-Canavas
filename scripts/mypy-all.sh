#!/usr/bin/env bash
# Runs `mypy --strict` over every first-party package/service.
#
# Why not one `mypy packages services` call: services/api, services/worker
# and services/scheduler each have their own top-level `app/` package by
# design (CLAUDE.md §2 — each service is run with its own directory as cwd,
# e.g. `cd services/api && uvicorn app.main:app`). Checking more than one of
# them in a single mypy invocation makes mypy see two files both claiming to
# be module `app.*` ("Source file found twice under different module
# names"). So each service is checked from within its own directory, and
# packages/* (which don't collide) are checked together from the repo root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

status=0

echo "== packages/contracts, packages/storage =="
uv run mypy packages/contracts/contracts packages/storage/storage || status=$?

echo "== migrations (task 8.1, repo-root-safe: no app.* import) =="
uv run mypy migrations || status=$?

echo "== ml/registry, ml/reward_model, ml/experiment_tracking (task 9.2/9.6/9.10) =="
uv run mypy ml/registry.py ml/reward_model ml/experiment_tracking.py || status=$?

# Checked separately from packages/*: mypy resolves the `from storage import
# ...` inside these test files differently once `tests` is scanned in the
# same invocation as packages/storage's own source, reintroducing the same
# "found twice under different module names" error the split above avoids.
#
# tests/unit/llm (and any future tests/unit/<service>) is excluded here —
# those import a service's `app.*`, which only resolves with that
# service's directory as cwd, same as scripts/test-all.sh.
echo "== tests (repo-root-safe subset) =="
uv run mypy tests/unit/contracts tests/unit/storage tests/integration || status=$?

for service in api worker scheduler; do
  echo "== services/${service} =="
  (cd "services/${service}" && uv run mypy app) || status=$?
done

echo "== tests/unit/llm (needs services/worker as cwd) =="
(cd services/worker && uv run mypy ../../tests/unit/llm) || status=$?

echo "== tests/unit/scheduler (needs services/scheduler as cwd) =="
(cd services/scheduler && uv run mypy ../../tests/unit/scheduler) || status=$?

echo "== tests/unit/api (needs services/api as cwd) =="
(cd services/api && uv run mypy ../../tests/unit/api) || status=$?

echo "== tests/e2e (needs services/api as cwd) =="
(cd services/api && uv run mypy ../../tests/e2e) || status=$?

# ml/evals/extractor.py bootstraps services/worker onto sys.path itself
# (see its module docstring) — same cwd requirement for the same reason.
# ml/identity/faces.py doesn't need app.* itself, but checking it here
# too (rather than a separate invocation) is harmless and keeps one home
# for "ml/ modules that will eventually need app.*" as identity/ grows
# (task 5.4's lookcard.py will call llm.routing the same way evals do).
echo "== ml/evals, ml/identity (needs services/worker as cwd) =="
(cd services/worker && uv run mypy ../../ml/evals ../../ml/identity) || status=$?

# tests/unit/ml imports ml.evals.* (needs repo root as a base — the
# nearest-ancestor-without-__init__.py walk from tests/unit/ml/*.py lands
# there) AND, transitively, app.* (needs services/worker as a base) — two
# bases at once, which neither directory's cwd gives on its own. MYPYPATH
# adds services/worker as a second base without cwd leaving the repo
# root, where `ml.*` still resolves via the ordinary __init__.py-chain walk.
echo "== tests/unit/ml (repo root + services/worker as a second base) =="
MYPYPATH=services/worker uv run mypy tests/unit/ml || status=$?

exit "$status"
