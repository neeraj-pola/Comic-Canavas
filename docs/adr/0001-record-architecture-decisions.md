# 0001. Record architecture decisions

Status: Accepted

## Context

CLAUDE.md fixes most technology choices up front (§1's decisions table),
but the build will still hit forks not already settled there — model
choice per role, prod database placement, prompt-optimization strategy.
Those choices need to survive past the PR that made them.

## Decision

Use lightweight Architecture Decision Records (Michael Nygard's format),
one per real fork, numbered sequentially in `docs/adr/`. CLAUDE.md §0.1
requires one "whenever choosing between alternatives that were discussed
in this file" — ten lines is enough; this file is the template for those.

## Consequences

Decisions already fixed in CLAUDE.md §1 don't get an ADR (reopening them
needs one instead, per §1's "do not reopen without an ADR"). Every ADR
promised elsewhere in CLAUDE.md (0003–0008) is a real gate: the phase that
names it isn't done until the ADR exists.
