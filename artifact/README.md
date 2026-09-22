# Comic Canvas — public demo

A static, look-but-don't-touch showcase of the real [Comic Canvas](../README.md) app. Same UI, same
components, same real generated images — just no live backend. Every button that would normally
call the real API instead shows a small modal: *"Nothing was changed. Clone the project and run it
with your own API keys to generate real comics of your own."*

## Why this exists, not the real app deployed as-is

The real app calls real, metered APIs (LLM text, image generation) for every comic it draws.
Deploying it publicly as-is would mean anyone visiting the site spends real money on the owner's
API keys — not appropriate for a personal project. This folder is the alternative: a genuine look
and feel of the product, backed by real output from actually running it, with zero ongoing cost.

## How it works

- **Same frontend, unmodified.** Every page and component under `app/`/`components/` is the real
  app's code, copied as-is. The UI/UX is identical.
- **`lib/api.ts` is the only thing swapped.** The real version calls a FastAPI backend; this version
  reads from `lib/demo-data.json` (a static export of real rows from the real local database) and
  serves real images from `public/demo/` (real generated strips/panels/candidates, resized and
  recompressed to keep this small). Every function keeps the real app's exact name and type
  signature, so no page needed to change.
- **Every mutation shows the notice modal** (`lib/demo-notice.tsx`) instead of doing anything —
  regenerating a panel, submitting a tap/rating/caption, training a character, saving settings,
  exporting/deleting your account. Reads (browsing days, the library, weekly recaps, the learning
  page, the cast page) are all real, static data.
- **Downloads work for real.** Strip/panel images are real static files, so browser "save image"
  and any in-app download button that just opens a real `/demo/...` URL both work genuinely.
- **Dates are anchored to the real dataset**, not the visitor's real calendar date — "Today"
  defaults to the latest real generated day (currently 2026-09-21), and Library/Weekly default to
  the month/week that dataset actually covers, so the very first thing anyone sees is real content,
  not an empty state.

## Running locally

```bash
pnpm install
pnpm dev
```

No `.env` file, no database, no API keys needed — everything is static.

## Deploying to Vercel

This is a normal Next.js app with no server-side dependencies (no database, no env vars required).
Point a new Vercel project at this repo with **Root Directory** set to `artifact/`, or push this
folder to its own repo and import that. Either way, `pnpm install && pnpm build` is all it needs.

## Regenerating the demo data

The dataset (`lib/demo-data.json` + `public/demo/`) was built once from the real local database and
real generated images by `scripts/build_demo_artifact.py` at the repo root. Re-run it (from the
repo root, with a real `DATABASE_URL` and real generated days in `.data/`) any time you want to
refresh this demo with more recent real output:

```bash
uv run --project services/worker python scripts/build_demo_artifact.py
```

It queries every real day with a composed strip, copies+resizes the real images it references, and
writes everything into this folder. Safe to re-run; it skips images it's already copied.
