"""add image_prompts table

`ImagePrompt` was pure per-run pipeline data, never persisted
relationally, so `GET /days/{date}`'s `prompts` field was always `[]`.
Prompt-preference training (mapping which prompt produced a chosen vs.
rejected candidate) needs the actual prompt text to still exist after the
graph run finishes, which nothing kept.

One row per real generation attempt (`id`, not `(day_id, panel_id)`,
since a retry — the critic's identity-gate retry, or a manual
`/regenerate` — writes a new prompt for the same panel; only the last
attempt's candidates survive into the final `DayState.candidates` either
way, since a retry discards the failed batch entirely, so
`candidates.prompt_id` always points at the prompt that actually produced
it). `seed` is `BIGINT`, not `INT`, from the start — `nodes/prompts.py`'s
`seed_for_panel` derives seeds from 8 hex digits (up to ~4.3B), the same
overflow already found and fixed once for `candidates.seed` (migration
`fa6f0a01842d`).

Revision ID: 582963a9d783
Revises: fa6f0a01842d
Create Date: 2026-09-16 21:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "582963a9d783"
down_revision: str | Sequence[str] | None = "fa6f0a01842d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE image_prompts (
            id TEXT PRIMARY KEY,
            day_id TEXT NOT NULL,
            panel_id INT NOT NULL,
            generator TEXT NOT NULL,
            character_clause TEXT NOT NULL,
            environment_clause TEXT NOT NULL,
            positive TEXT NOT NULL,
            negative TEXT NOT NULL,
            seed BIGINT,
            guidance DOUBLE PRECISION,
            steps INT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            FOREIGN KEY (day_id, panel_id) REFERENCES panels(day_id, panel_id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX image_prompts_day_panel_idx ON image_prompts(day_id, panel_id)")
    op.execute("ALTER TABLE candidates ADD COLUMN prompt_id TEXT REFERENCES image_prompts(id)")


def downgrade() -> None:
    op.execute("ALTER TABLE candidates DROP COLUMN IF EXISTS prompt_id")
    op.execute("DROP TABLE IF EXISTS image_prompts CASCADE")
