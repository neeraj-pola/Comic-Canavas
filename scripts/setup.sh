#!/usr/bin/env bash
# One-command setup for macOS, Linux, and Windows via WSL2 or Git Bash.
#
# What this does NOT do, on purpose: install Postgres/Redis for you. Building
# pgvector from source and running a local Postgres/Redis is real, OS-specific
# work that a script can get subtly wrong (see README's own build commands).
# The realistic "works the same on every OS" path is a free hosted Postgres
# (Neon, native pgvector, no card) + a free hosted Redis (Upstash) — this
# script assumes you'll point DATABASE_URL/REDIS_URL at those, or at a local
# install you've already set up yourself per the README.
#
# What it does: installs uv (Python) and pnpm (Node) if missing, wires up
# .env, installs every workspace dependency, links apps/web/.env, and runs
# the database migration — the same steps `make install && make migrate`
# already do, just with checks and clear messages at each step.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd)"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
info() { printf '  %s\n' "$1"; }
warn() { printf '\033[33m! %s\033[0m\n' "$1"; }
die()  { printf '\033[31m✗ %s\033[0m\n' "$1"; exit 1; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$1"; }

bold "Comic Canvas setup"
info "repo: $REPO_ROOT"
echo

# ---------------------------------------------------------------------------
# 1. uv (Python 3.12 + the whole workspace's virtualenv)
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  bold "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH yet — open a new terminal (or 'source ~/.bashrc'/'~/.zshrc') and re-run this script."
fi
ok "uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 2. Node + pnpm
# ---------------------------------------------------------------------------
if ! command -v node >/dev/null 2>&1; then
  die "Node 20+ isn't installed. Install it from https://nodejs.org (or via nvm), then re-run this script."
fi
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
[ "$NODE_MAJOR" -ge 20 ] || die "Node $NODE_MAJOR found, but Node 20+ is required."
ok "node: $(node --version)"

if ! command -v pnpm >/dev/null 2>&1; then
  bold "Enabling pnpm via corepack..."
  corepack enable 2>/dev/null || npm install -g corepack
  corepack prepare pnpm@10.16.1 --activate
fi
ok "pnpm: $(pnpm --version)"
echo

# ---------------------------------------------------------------------------
# 3. .env
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
  cp .env.example .env
  ok "created .env from .env.example"
else
  info ".env already exists — leaving it alone"
fi

# DATABASE_URL/REDIS_URL are commented out with no default in .env.example
# (make dev/this script refuse to guess) — check for an actual, live value.
has_value() {
  grep -qE "^${1}=.+" .env 2>/dev/null
}

if ! has_value DATABASE_URL || ! has_value REDIS_URL; then
  echo
  warn "DATABASE_URL and/or REDIS_URL aren't set in .env yet — stopping here."
  echo
  bold "The fastest path on any OS (no local Postgres/Redis install):"
  info "1. Postgres:  https://neon.com — free tier, pgvector built in. Copy its connection string."
  info "2. Redis:     https://upstash.com — free tier. Copy its REDIS_URL (rediss://...)."
  info "3. Open .env and set DATABASE_URL=... and REDIS_URL=... to those two values."
  info "4. Also set ANTHROPIC_API_KEY and/or OPENAI_API_KEY if you want real generation"
  info "   (leave IMAGE_PROVIDER=mock and every LLM_* role as-is to run with zero cost)."
  echo
  info "Then re-run: ./scripts/setup.sh"
  exit 1
fi
ok "DATABASE_URL and REDIS_URL are set"
echo

# ---------------------------------------------------------------------------
# 4. Install dependencies
# ---------------------------------------------------------------------------
bold "Installing Python dependencies (uv sync --all-packages)..."
uv sync --all-packages
echo
bold "Installing Node dependencies (pnpm install)..."
pnpm install
echo

# Next.js only reads .env* from its own directory (unlike the Python
# services, which walk up to find the repo-root .env) — link it in.
if [ ! -e apps/web/.env ]; then
  ln -s ../../.env apps/web/.env
  ok "linked apps/web/.env -> ../../.env"
fi

# ---------------------------------------------------------------------------
# 5. Database migration
# ---------------------------------------------------------------------------
bold "Running migrations (alembic upgrade head)..."
if ! uv run alembic upgrade head; then
  die "Migration failed — check DATABASE_URL in .env (is the database reachable, does it have the vector extension enabled?)."
fi
ok "database is up to date"
echo

bold "Done. Start the app with two terminals:"
info "make dev             # api + worker + scheduler"
info "cd apps/web && pnpm dev   # frontend, in a second terminal"
echo
info "Then open the URL pnpm dev prints (usually http://localhost:3000)."
