# One-command setup for native Windows (PowerShell), no WSL required.
#
# What this does NOT do, on purpose: install Postgres/Redis for you, and it
# doesn't use `make` (not on Windows by default). Building pgvector from
# source and running a native Windows Postgres/Redis is real, fiddly,
# platform-specific work. The realistic "just works" path on Windows is a
# free hosted Postgres (Neon, native pgvector, no card) + a free hosted
# Redis (Upstash) — point DATABASE_URL/REDIS_URL in .env at those.
#
# What it does: installs uv (Python) and pnpm (Node) if missing, wires up
# .env, installs every workspace dependency, links apps/web/.env, and runs
# the database migration.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$RepoRoot = Get-Location

function Write-Bold($msg) { Write-Host $msg -ForegroundColor White }
function Write-Info($msg) { Write-Host "  $msg" }
function Write-Ok($msg)   { Write-Host "OK $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "! $msg" -ForegroundColor Yellow }
function Die($msg)        { Write-Host "X $msg" -ForegroundColor Red; exit 1 }

Write-Bold "Comic Canvas setup"
Write-Info "repo: $RepoRoot"
Write-Host ""

# ---------------------------------------------------------------------------
# 1. uv (Python 3.12 + the whole workspace's virtualenv)
# ---------------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Bold "Installing uv..."
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Die "uv installed but not on PATH yet - open a new PowerShell window and re-run this script."
    }
}
Write-Ok "uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 2. Node + pnpm
# ---------------------------------------------------------------------------
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Die "Node 20+ isn't installed. Install it from https://nodejs.org, then re-run this script."
}
$nodeMajor = [int]((node -p "process.versions.node.split('.')[0]"))
if ($nodeMajor -lt 20) { Die "Node $nodeMajor found, but Node 20+ is required." }
Write-Ok "node: $(node --version)"

if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
    Write-Bold "Enabling pnpm via corepack..."
    corepack enable
    corepack prepare pnpm@10.16.1 --activate
}
Write-Ok "pnpm: $(pnpm --version)"
Write-Host ""

# ---------------------------------------------------------------------------
# 3. .env
# ---------------------------------------------------------------------------
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Ok "created .env from .env.example"
} else {
    Write-Info ".env already exists - leaving it alone"
}

function Has-Value($name) {
    $line = Select-String -Path .env -Pattern "^$name=.+" -ErrorAction SilentlyContinue
    return $null -ne $line
}

if (-not (Has-Value "DATABASE_URL") -or -not (Has-Value "REDIS_URL")) {
    Write-Host ""
    Write-Warn "DATABASE_URL and/or REDIS_URL aren't set in .env yet - stopping here."
    Write-Host ""
    Write-Bold "The fastest path on Windows (no native Postgres/Redis install):"
    Write-Info "1. Postgres:  https://neon.com - free tier, pgvector built in. Copy its connection string."
    Write-Info "2. Redis:     https://upstash.com - free tier. Copy its REDIS_URL (rediss://...)."
    Write-Info "3. Open .env and set DATABASE_URL=... and REDIS_URL=... to those two values."
    Write-Info "4. Also set ANTHROPIC_API_KEY and/or OPENAI_API_KEY if you want real generation"
    Write-Info "   (leave IMAGE_PROVIDER=mock and every LLM_* role as-is to run with zero cost)."
    Write-Host ""
    Write-Info "Then re-run: .\scripts\setup.ps1"
    exit 1
}
Write-Ok "DATABASE_URL and REDIS_URL are set"
Write-Host ""

# ---------------------------------------------------------------------------
# 4. Install dependencies
# ---------------------------------------------------------------------------
Write-Bold "Installing Python dependencies (uv sync --all-packages)..."
uv sync --all-packages
Write-Host ""
Write-Bold "Installing Node dependencies (pnpm install)..."
pnpm install
Write-Host ""

# Next.js only reads .env* from its own directory. Windows symlinks need
# admin/Developer Mode, so this copies instead - re-run this script after
# editing the root .env to refresh apps/web/.env.
if (-not (Test-Path apps/web/.env)) {
    Copy-Item .env apps/web/.env
    Write-Ok "copied .env -> apps/web/.env (re-run this script after editing the root .env)"
}

# ---------------------------------------------------------------------------
# 5. Database migration
# ---------------------------------------------------------------------------
Write-Bold "Running migrations (alembic upgrade head)..."
uv run alembic upgrade head
if ($LASTEXITCODE -ne 0) {
    Die "Migration failed - check DATABASE_URL in .env (is the database reachable, does it have the vector extension enabled?)."
}
Write-Ok "database is up to date"
Write-Host ""

Write-Bold "Done. Start the app with two PowerShell windows:"
Write-Info "uv run honcho start -f Procfile.dev   # api + worker + scheduler (make isn't used on Windows)"
Write-Info "cd apps/web; pnpm dev                 # frontend, in a second window"
Write-Host ""
Write-Info "Then open the URL pnpm dev prints (usually http://localhost:3000)."
