"""candidate embeddings

The personal ranking model learns from an image embedding (DINOv2) on top
of the named critic signals. Stored per candidate when it is scored, so
retraining after a pick never has to re-read and re-embed old images.

Revision ID: a7d3e1f8b520
Revises: f6c2d8a94e13
Create Date: 2026-09-21 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "a7d3e1f8b520"
down_revision: str | Sequence[str] | None = "f6c2d8a94e13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE candidates ADD COLUMN embedding JSONB")
    # The projection (mean + top components) the ranking model was trained with.
    op.execute("ALTER TABLE preference_models ADD COLUMN embed JSONB NOT NULL DEFAULT '{}'::jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE preference_models DROP COLUMN embed")
    op.execute("ALTER TABLE candidates DROP COLUMN embedding")
