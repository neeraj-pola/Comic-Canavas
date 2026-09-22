"""add kind to days (daily vs weekly recap)

The weekly recap was stored as a synthetic `days` row dated the week's Sunday,
but `days` is UNIQUE (user_id, date) — so for anyone with an ordinary diary
entry for that Sunday (the normal case) the recap either failed to save or,
worse, would have shown up as a second diary entry. `kind` separates them:
uniqueness is now (user_id, date, kind), and every query that means "the
person's diary" reads `kind = 'daily'`.

Downgrade deletes weekly rows first (derived data, regenerable from the daily
rows) because the old unique constraint can't hold both.

Revision ID: e5b9c3d1a7f2
Revises: d4a8f2c61b73
Create Date: 2026-09-20 12:30:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "e5b9c3d1a7f2"
down_revision: str | Sequence[str] | None = "d4a8f2c61b73"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE days ADD COLUMN kind TEXT NOT NULL DEFAULT 'daily' "
        "CHECK (kind IN ('daily', 'weekly'))"
    )
    op.execute("UPDATE days SET kind = 'weekly' WHERE job_id LIKE '%:weekly:%'")
    op.execute("ALTER TABLE days DROP CONSTRAINT days_user_id_date_key")
    op.execute(
        "ALTER TABLE days ADD CONSTRAINT days_user_id_date_kind_key UNIQUE (user_id, date, kind)"
    )


def downgrade() -> None:
    op.execute("DELETE FROM days WHERE kind = 'weekly'")
    op.execute("ALTER TABLE days DROP CONSTRAINT days_user_id_date_kind_key")
    op.execute("ALTER TABLE days ADD CONSTRAINT days_user_id_date_key UNIQUE (user_id, date)")
    op.execute("ALTER TABLE days DROP COLUMN kind")
