"""add gender_term to people

`LookCard.gender_term` is deliberately not vision-inferred — guessing
gender from photos risks misgendering and isn't more reliable than just
asking the person — and `ml/identity/lookcard.py`'s `build_look_card`
already accepts a `gender_term` parameter to carry a user-provided value
through. But no caller ever collected or passed one, so it silently
stayed empty for every character trained, and every panel's
`character_clause` fell back to the neutral "a person" instead of a real
anchor. `people` needs somewhere to hold the answer once asked (the setup
wizard) so it survives to be read back when training actually runs.

Revision ID: 8d75dad43698
Revises: 582963a9d783
Create Date: 2026-09-18 19:55:31.884401

"""

from collections.abc import Sequence

from alembic import op

revision: str = "8d75dad43698"
down_revision: str | Sequence[str] | None = "582963a9d783"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE people ADD COLUMN gender_term TEXT NOT NULL DEFAULT ''")


def downgrade() -> None:
    op.execute("ALTER TABLE people DROP COLUMN gender_term")
