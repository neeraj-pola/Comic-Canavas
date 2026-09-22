# 0002. Backend first, deploy last

Status: Accepted

## Context

The product needs a working pipeline (extractor → script → prompts →
images → critic → compose) far more than it needs a hosted URL early. The
pipeline is also where the cost and correctness risk lives (LLM/ASR/image
spend, identity consistency, eval thresholds) — frontend and hosting risk
are comparatively well-trodden.

## Decision

Phases 1–9 build and test the full backend locally against a Supabase
dev Postgres, local Redis, and a filesystem storage adapter, with mock
image generation standing in until Phase 4. Phase 10 wires the
already-designed frontend to that backend. Phase 12 deploys once, at the
end — no Docker, no staging environment, no mid-build hosting accounts
(Railway/Vercel/R2/Modal) until then.

## Consequences

Every phase gate before 12 is verifiable with `make dev` / `make test` on
a laptop — no deploy step can hide a broken contract. The cost: nothing
is publicly reachable, and integration bugs specific to the real hosting
environment (Railway/Vercel network behavior, R2 CORS, cold starts) are
only found in Phase 12, right before the deadline they're graded against.
