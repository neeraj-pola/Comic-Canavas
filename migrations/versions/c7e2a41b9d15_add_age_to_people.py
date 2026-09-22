"""add age to people

Asks the person's age up front, alongside gender, for more relatability in
the comics. `LookCard.age_descriptor` already exists (test renders skewed
younger than the photos without an age anchor) and is deliberately
user-provided, not vision-guessed — but nothing ever collected it. The
setup wizard now asks; the worker turns it into "{age}-year-old" on the
look card, so every panel's `character_clause` reads e.g.
"a 34-year-old man, ...".

Revision ID: c7e2a41b9d15
Revises: b3c91d7a4e20
Create Date: 2026-09-19 19:20:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "c7e2a41b9d15"
down_revision: str | Sequence[str] | None = "b3c91d7a4e20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE people ADD COLUMN age INT CHECK (age BETWEEN 1 AND 120)")


def downgrade() -> None:
    op.execute("ALTER TABLE people DROP COLUMN age")
