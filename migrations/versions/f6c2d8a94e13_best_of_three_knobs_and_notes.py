"""best-of-3 taps, knob model, taste notes, personalise switch

A quick tap is now a pick of the favourite of a panel's three options
(Plackett-Luce), so the rows one tap writes to `image_pairs` (chosen vs
each other option) share a `tap_id`. The preference model gains a second,
smaller model over the three prompt knobs (`knobs`: posterior, the
person's default level per knob, and the curves the Learning page draws)
and a short written summary of their taste (`notes`). Snapshots keep
every option the person passed on and whether they picked the default.
`settings.personalise` is the off switch for all of it.

Revision ID: f6c2d8a94e13
Revises: e5b9c3d1a7f2
Create Date: 2026-09-20 22:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "f6c2d8a94e13"
down_revision: str | Sequence[str] | None = "e5b9c3d1a7f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE image_pairs ADD COLUMN tap_id TEXT")
    op.execute(
        "ALTER TABLE preference_models "
        "ADD COLUMN knobs JSONB NOT NULL DEFAULT '{}'::jsonb, "
        "ADD COLUMN notes JSONB NOT NULL DEFAULT '{}'::jsonb"
    )
    op.execute(
        "ALTER TABLE preference_snapshots "
        "ADD COLUMN rejected_urls JSONB NOT NULL DEFAULT '[]'::jsonb, "
        "ADD COLUMN default_picked BOOLEAN"
    )
    op.execute("ALTER TABLE settings ADD COLUMN personalise BOOLEAN NOT NULL DEFAULT true")


def downgrade() -> None:
    op.execute("ALTER TABLE settings DROP COLUMN personalise")
    op.execute(
        "ALTER TABLE preference_snapshots DROP COLUMN default_picked, DROP COLUMN rejected_urls"
    )
    op.execute("ALTER TABLE preference_models DROP COLUMN notes, DROP COLUMN knobs")
    op.execute("ALTER TABLE image_pairs DROP COLUMN tap_id")
