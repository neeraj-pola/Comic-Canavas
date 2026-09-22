"""image ratings

A pick says which of three images is best; it can never say whether any
of them is good. A rating (0 off, 1 ok, 2 great) of a single image does,
so the personal models learn from both. One row per rating given (a
changed mind is a new row; the latest per image counts).

Revision ID: b8e4f2a1c9d7
Revises: a7d3e1f8b520
Create Date: 2026-09-21 18:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "b8e4f2a1c9d7"
down_revision: str | Sequence[str] | None = "a7d3e1f8b520"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE image_ratings (
            id BIGSERIAL PRIMARY KEY,
            day_id TEXT NOT NULL REFERENCES days(job_id) ON DELETE CASCADE,
            panel_id INT NOT NULL,
            url TEXT NOT NULL,
            rating SMALLINT NOT NULL CHECK (rating BETWEEN 0 AND 2),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX image_ratings_day ON image_ratings (day_id, panel_id)")


def downgrade() -> None:
    op.execute("DROP TABLE image_ratings")
