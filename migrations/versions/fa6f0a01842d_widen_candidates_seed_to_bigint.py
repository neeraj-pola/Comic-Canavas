"""widen candidates.seed to bigint

`seed_for_panel` (`nodes/prompts.py`) derives seeds from 8 hex digits of a
sha256 hash — up to 0xFFFFFFFF (4,294,967,295) — but Postgres `INT` is a
signed 32-bit integer (max 2,147,483,647), so a seed in the upper half of
that range overflows with `psycopg.errors.NumericValueOutOfRange`.
`BIGINT` comfortably covers it with no application-code change needed.

Revision ID: fa6f0a01842d
Revises: af5692fa786e
Create Date: 2026-09-16 12:06:48.148286

"""

from collections.abc import Sequence

from alembic import op

revision: str = "fa6f0a01842d"
down_revision: str | Sequence[str] | None = "af5692fa786e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE candidates ALTER COLUMN seed TYPE BIGINT")


def downgrade() -> None:
    op.execute("ALTER TABLE candidates ALTER COLUMN seed TYPE INT")
