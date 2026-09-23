# Comic Canvas

A diary that draws itself. Type your day → a four-panel comic with a consistent drawn character
of you in it → two quick taps a day teach it your taste, and it slowly leans your daily strips
toward what you actually pick.

This README covers what you need to clone it, fill in your own keys, and run it.

## What this is

- **A FastAPI backend + arq worker running a LangGraph pipeline**: your diary text → extracted
  beats → a 4-panel script → image prompts → generated candidates → a critic that scores them →
  a composed comic strip.
- **A Next.js frontend** (Today, Library, Weekly, Cast, Style, Learning, Account) wired to that
  backend.
- **A personal preference model** that learns from which of 3 drawn options you pick each panel
  and how you rate the one in your strip, and nudges future prompts (warmth, framing, expression)
  toward what you keep choosing.
- **Single-user, local-only, no login.** There is no sign-up flow and no per-user accounts — the
  whole app is built to run on your own machine, against your own local Postgres, for one person.

## Setup

**Fastest path (macOS, Linux, or Windows):**

```bash
git clone <this-repo-url> comic-canvas && cd comic-canvas
./scripts/setup.sh   # macOS / Linux / Windows via WSL2 or Git Bash
```

```powershell
git clone <this-repo-url> comic-canvas; cd comic-canvas
.\scripts\setup.ps1   # native Windows PowerShell (no WSL needed)
```

Either script installs `uv`/`pnpm` if missing, creates `.env` from `.env.example`, and — the
first time you run it — stops with a short checklist if `DATABASE_URL`/`REDIS_URL` aren't filled
in yet (pointing at [Neon](https://neon.com) + [Upstash](https://upstash.com)'s free tiers, since
that's the one path that's identical on every OS and needs no native Postgres/pgvector/Redis
build). Fill those two lines in, add an LLM key if you want real generation, and re-run the
script — it finishes with the exact single command to start the app. Everything below this is what
the script automates, for anyone who wants to do it by hand or understand what it's doing.

You need:

1. **Python 3.12 via [uv](https://docs.astral.sh/uv/)** — `uv sync --all-packages` creates one
   shared virtualenv (`.venv`) for every Python package in the workspace.
2. **Node 20+ via [pnpm](https://pnpm.io/)** — `pnpm install` at the repo root installs both the
   root tooling and `apps/web`.
3. **Redis**, running locally — `brew install redis && brew services start redis` (or an
   [Upstash](https://upstash.com/) free-tier `REDIS_URL`).
4. **A Postgres 15+ with the `vector` extension enabled.** Two real options:
   - **A native local Postgres** — `brew install postgresql@15 && brew services start postgresql@15`,
     then build [pgvector](https://github.com/pgvector/pgvector) from source against it (Homebrew's
     pgvector bottle only targets pg 17/18): `git clone --depth 1 --branch v0.8.0
     https://github.com/pgvector/pgvector.git && cd pgvector && PG_CONFIG=$(brew --prefix
     postgresql@15)/bin/pg_config make -j4 && PG_CONFIG=$(brew --prefix postgresql@15)/bin/pg_config
     make install`. Then `createuser comiccanvas && createdb -O comiccanvas comiccanvas && psql -d
     comiccanvas -c "CREATE EXTENSION vector;"`.
   - **[Neon](https://neon.com)** (free tier, no card required, native pgvector) — create a
     project and copy its connection string.
5. **At least one LLM provider key** (Anthropic and/or OpenAI) to get a real strip end to end.
   Without one, the backend still runs and every test still passes against mock providers —
   you just won't get real generated text or images.

Then:

```bash
cp .env.example .env
# open .env and fill in, at minimum:
#   DATABASE_URL, REDIS_URL              — from steps 4/3 above; make dev refuses to start without them
#   ANTHROPIC_API_KEY and/or OPENAI_API_KEY   — for real script/beat/image-prompt generation
#   FAL_KEY                              — for real image generation (IMAGE_PROVIDER=flux_kontext);
#                                          leave IMAGE_PROVIDER=mock to skip this and every image cost
# NEXT_PUBLIC_DEV_USER_ID / NEXT_PUBLIC_DEV_USER_NAME can be left as the generic defaults —
# there's no login, so this is just which rows in your own database the app reads.

make install     # uv sync --all-packages && pnpm install && symlinks apps/web/.env -> ../../.env
                 # (Next.js only reads .env files from its own directory)
make migrate     # creates every table in your database (alembic upgrade head)
make start       # single command: api + worker + scheduler + web, all in one terminal
```

`GET http://localhost:8000/health` should return `{"status": "ok", "db": true, "redis": true}`.
Open `http://localhost:3000` (`make start` prints the real URL) — you land straight on the
onboarding wizard the first time (no character yet), or Today once one exists.

Prefer two separate terminals (e.g. to keep frontend logs out of the backend's), or want to run
just the backend, `make dev` (backend only, api + worker + scheduler) and, separately, `pnpm dev`
(frontend only) still work exactly as before, see [Commands](#commands) below.

### First run: building your character

Before any comic can be drawn you need an approved character: **Cast → add photos → Train**
(a real, paid call to fal.ai once you confirm — a few dollars; the cost is shown before you
confirm) **→ pick your favourite of the generated candidates**. After that, Today generates real
4-panel strips from whatever you type.

If you don't want to spend anything yet, set `IMAGE_PROVIDER=mock` and every LLM role to
`mock:mock` in `.env` — the full pipeline runs end to end with deterministic placeholder images
and zero real cost, which is also exactly what `make test` uses.

## Commands

```bash
make start         # api + worker + scheduler + web, all one command, one terminal (honcho, Procfile.full)
make dev           # api + worker + scheduler only (honcho, Procfile.dev), backend only
pnpm dev           # the frontend only, run separately from `make dev` (see Setup above)
make test          # unit + integration, mock providers only, zero network calls
make migrate       # alembic upgrade head
make seed          # loads fixtures: one user, 7 days, cast of 3
make lint          # ruff + mypy --strict + eslint
```

`make help` prints the full list from the Makefile. `make test-live` and `make eval` run real
golden-set evals against real provider keys and print real spend before you confirm anything.

## Repository layout

- `packages/contracts`, `packages/storage` — shared Pydantic contracts and the storage interface
  (`LocalFileStorage` for dev, `R2Storage` for a real deployment).
- `services/api`, `services/worker`, `services/scheduler` — the three Python services, each with
  its own `app/` package.
- `apps/web` — the Next.js frontend.
- `ml/` — identity/character pipeline, evals, the preference-learning model's offline tooling.
- `migrations/` — Alembic, one real migration per schema change, in order.


## License

[Apache License 2.0](./LICENSE).
