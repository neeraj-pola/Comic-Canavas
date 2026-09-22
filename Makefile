.DEFAULT_GOAL := help
.PHONY: help install dev test test-live eval lint migrate seed clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install Python (uv) and Node (pnpm) dependencies for the whole workspace
	uv sync --all-packages
	pnpm install
	@# Next.js only reads .env* files from its own app directory (task
	@# 10.3, real bug found live: "@clerk/nextjs: Missing publishableKey"
	@# even with the repo-root .env fully populated) — unlike the Python
	@# services, which all walk up via find_dotenv(usecwd=True). A
	@# symlink is gitignored (matches the .env pattern), so it's recreated
	@# here on every install rather than committed.
	@[ -L apps/web/.env ] || [ -f apps/web/.env ] || ln -s ../../.env apps/web/.env

dev: ## Run api (uvicorn --reload) + worker (arq) + scheduler via honcho (Procfile.dev)
	uv run honcho start -f Procfile.dev

test: ## Unit + integration tests, mock providers only (zero network calls)
	./scripts/test-all.sh

test-live: ## Gate 1: extractor fixture on both real providers, prints spend (needs API keys)
	uv run python -m ml.evals.provider_parity

eval: ## ml/evals: extractor recall, script compliance/judge, critic accuracy, reward-model held-out (Phase 2+) — costs money
	uv run python ml/evals/extractor.py
	uv run python ml/evals/script.py
	@# critic accuracy (Phase 6) and reward-model held-out (Phase 9) join here as those land.

lint: ## ruff + mypy --strict (Python) + eslint + prettier (TS)
	uv run ruff check .
	uv run ruff format --check .
	./scripts/mypy-all.sh
	pnpm lint
	pnpm format

migrate: ## alembic upgrade head (task 8.1)
	uv run alembic upgrade head

seed: ## Load fixtures: one user, 7 days, cast of 3 (task 8.1)
	uv run python scripts/seed.py

clean: ## Remove caches and local dev data (.venv and node_modules are left alone)
	find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' \) -not -path '*/.venv/*' -not -path '*/node_modules/*' -exec rm -rf {} +
	rm -rf .data/*
