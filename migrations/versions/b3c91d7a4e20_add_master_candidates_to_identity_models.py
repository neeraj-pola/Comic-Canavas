"""add master_candidates to identity_models

Instead of `train_character_job` auto-approving the top-ranked master
candidate (and discarding the rest), the person picks one of the top few
themselves ("which one feels like you"). The ranked candidates need to
survive between the training job finishing and the person choosing, so
they live here as JSONB: `[{id, url, seed, rank_score, style_score,
checklist_score, style_card_version}]`, cleared once one is approved.

Revision ID: b3c91d7a4e20
Revises: 8d75dad43698
Create Date: 2026-09-19 18:10:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "b3c91d7a4e20"
down_revision: str | Sequence[str] | None = "8d75dad43698"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE identity_models ADD COLUMN master_candidates JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE identity_models DROP COLUMN master_candidates")
