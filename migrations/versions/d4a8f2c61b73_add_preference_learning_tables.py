"""add preference learning tables

The app learns each person's taste from their A/B taps (Bayesian
Bradley-Terry over interpretable image features) and shows that learning
on the Learning page. `preference_models` holds the current posterior
(mean, covariance, feature scales) plus whether it is active and what it
leans toward; `preference_snapshots` holds one row per tap — the model's
held-out prediction for that tap and the weights after learning from it —
which is exactly what the charts replay.

Revision ID: d4a8f2c61b73
Revises: c7e2a41b9d15
Create Date: 2026-09-19 21:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "d4a8f2c61b73"
down_revision: str | Sequence[str] | None = "c7e2a41b9d15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE preference_models (
            user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            features JSONB NOT NULL,
            mu JSONB NOT NULL,
            cov JSONB NOT NULL,
            scales JSONB NOT NULL,
            n_taps INT NOT NULL,
            active BOOLEAN NOT NULL DEFAULT false,
            lean JSONB NOT NULL DEFAULT '{}'::jsonb,
            z JSONB NOT NULL DEFAULT '{}'::jsonb,
            accuracy JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE preference_snapshots (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            tap_index INT NOT NULL,
            day_id TEXT,
            panel_id INT,
            chosen_url TEXT,
            rejected_url TEXT,
            prob DOUBLE PRECISION NOT NULL,
            correct DOUBLE PRECISION NOT NULL,
            hand_correct DOUBLE PRECISION NOT NULL,
            mu JSONB NOT NULL,
            sd JSONB NOT NULL,
            delta JSONB NOT NULL,
            axis INT,
            tapped_at TIMESTAMPTZ,
            UNIQUE (user_id, tap_index)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE preference_snapshots")
    op.execute("DROP TABLE preference_models")
