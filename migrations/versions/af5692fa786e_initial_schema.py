"""initial schema — every real table the app's Pydantic contracts need.

Real, raw-SQL migration (no ORM anywhere in this codebase, `op.execute`
throughout, not `op.create_table`) creating `users`, `people`,
`person_photos`, `identity_models`, `jobs`, `days`, `beats`, `panels`,
`candidates`, `image_pairs`, `caption_pairs`, `thumbs`, `settings`,
`checkpoints`, `training_runs`, `cost_events`, built from the real
Pydantic contracts they mirror (`contracts.day`, `contracts.identity`,
`contracts.feedback`, `app.llm.cost.CostEvent`,
`ml.identity.model.IdentityModel`), not guessed columns.

This migration does not touch or consolidate `memory_beats`/
`memory_people`/`memory_places` (runtime-bootstrapped tables, still
created idempotently by `app.db.ensure_schema` on every connection) —
this migration creates a new, separate `beats` table instead (relational
and FK'd to `days`, for the API layer's own querying needs), deliberately
leaving `memory_beats` (embedding-oriented, used internally by
`app.memory.PostgresMemoryStore` for semantic search) untouched. Two
tables with overlapping data for now, real and documented, not silently
duplicated — reconciling them is real, non-trivial surgery on working
code, left for a dedicated future pass.

`Beat.id` and `Panel.id` are only unique within one day (extractor/
script-writer numbering restarts "b1"/panel 1 every day), so `beats`/
`panels` here use `(day_id, beat_id)`/`(day_id, panel_id)` composite
primary keys from the start, not a bare `beat_id`/`panel_id` column.

Revision ID: af5692fa786e
Revises:
Create Date: 2026-09-16 11:00:10.595366

"""

from collections.abc import Sequence

from alembic import op

revision: str = "af5692fa786e"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            email TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY,  -- matches DayState.job_id for daily jobs
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            kind TEXT NOT NULL DEFAULT 'daily',  -- daily|weekly|export|training
            status TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
            events JSONB NOT NULL DEFAULT '[]',  -- per-node timing/versions
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE people (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            -- consent fields for recurring characters, not built yet
            uploaded_by TEXT REFERENCES users(id),
            consent_at TIMESTAMPTZ,
            revoked_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE person_photos (
            id BIGSERIAL PRIMARY KEY,
            person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
            storage_key TEXT NOT NULL,
            status TEXT,  -- usable|blur|duplicate|no_face|multi_face
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # Mirrors ml/identity/model.py's real IdentityModel field-for-field —
    # embedding is ArcFace's real 512-d vector; look_card is stored whole
    # as JSONB (read/written as one unit everywhere it's used, not
    # exploded into columns).
    op.execute(
        """
        CREATE TABLE identity_models (
            person_id TEXT PRIMARY KEY REFERENCES people(id) ON DELETE CASCADE,
            embedding vector(512),
            source_photo_count INT,
            mean_self_cosine DOUBLE PRECISION,
            look_card JSONB,
            leonardo_ref_ids TEXT[] NOT NULL DEFAULT '{}',
            flux_lora_url TEXT,
            trigger_token TEXT,
            master_path TEXT,
            master_approved_at TIMESTAMPTZ,
            master_candidate_id TEXT,
            master_seed INT,
            style_card_version TEXT,
            chip_rounds INT NOT NULL DEFAULT 0,
            sheet_paths TEXT[] NOT NULL DEFAULT '{}',
            sheet_generated_at TIMESTAMPTZ,
            generation_path TEXT,
            lora_checkpoint TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # Mirrors contracts.day.DayState's own fields for exactly what's
    # persisted per day (candidates/prompts live in their own tables
    # below).
    op.execute(
        """
        CREATE TABLE days (
            job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            date DATE NOT NULL,
            source TEXT NOT NULL,  -- text|audio
            text TEXT,
            audio_url TEXT,
            transcript TEXT,
            mood TEXT,
            quiet_day BOOLEAN,
            strip_url TEXT,
            story_url TEXT,
            layout JSONB,
            cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
            errors TEXT[] NOT NULL DEFAULT '{}',
            versions JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, date)
        )
        """
    )

    # beat_id is only unique within one day, so the primary key is composite.
    op.execute(
        """
        CREATE TABLE beats (
            day_id TEXT NOT NULL REFERENCES days(job_id) ON DELETE CASCADE,
            beat_id TEXT NOT NULL,
            time TEXT NOT NULL,
            place TEXT NOT NULL,
            place_detail TEXT,
            event TEXT NOT NULL,
            emotion TEXT NOT NULL,
            people TEXT[] NOT NULL DEFAULT '{}',
            objects TEXT[] NOT NULL DEFAULT '{}',
            importance DOUBLE PRECISION NOT NULL,
            humor DOUBLE PRECISION NOT NULL,
            quote TEXT,
            PRIMARY KEY (day_id, beat_id)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE panels (
            day_id TEXT NOT NULL REFERENCES days(job_id) ON DELETE CASCADE,
            panel_id INT NOT NULL,
            beat_id TEXT,
            place TEXT NOT NULL,
            time_of_day TEXT NOT NULL,
            expression TEXT NOT NULL,
            action TEXT NOT NULL,
            framing TEXT NOT NULL,
            caption_a TEXT NOT NULL,
            caption_b TEXT NOT NULL DEFAULT '',
            bubble TEXT NOT NULL DEFAULT '',
            cast_ids TEXT[] NOT NULL DEFAULT '{}',  -- "cast" is a reserved SQL word
            PRIMARY KEY (day_id, panel_id),
            FOREIGN KEY (day_id, beat_id) REFERENCES beats(day_id, beat_id)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE candidates (
            id TEXT PRIMARY KEY,
            day_id TEXT NOT NULL,
            panel_id INT NOT NULL,
            url TEXT NOT NULL,
            seed INT NOT NULL,
            scores JSONB NOT NULL DEFAULT '{}',
            face_box INT[],
            chosen BOOLEAN NOT NULL DEFAULT false,
            rejected_reason TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            FOREIGN KEY (day_id, panel_id) REFERENCES panels(day_id, panel_id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX candidates_day_panel_idx ON candidates(day_id, panel_id)")

    op.execute(
        """
        CREATE TABLE image_pairs (
            id BIGSERIAL PRIMARY KEY,
            day_id TEXT NOT NULL REFERENCES days(job_id) ON DELETE CASCADE,
            panel_id INT NOT NULL,
            chosen TEXT NOT NULL,
            rejected TEXT NOT NULL,
            source TEXT NOT NULL,  -- ab|regenerate
            checkpoint_id TEXT,
            versions JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE caption_pairs (
            id BIGSERIAL PRIMARY KEY,
            day_id TEXT NOT NULL REFERENCES days(job_id) ON DELETE CASCADE,
            panel_id INT NOT NULL,
            original TEXT NOT NULL,
            edited TEXT NOT NULL,
            checkpoint_id TEXT,
            versions JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE thumbs (
            id BIGSERIAL PRIMARY KEY,
            day_id TEXT NOT NULL REFERENCES days(job_id) ON DELETE CASCADE,
            panel_id INT NOT NULL,
            value SMALLINT NOT NULL CHECK (value IN (1, -1)),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE settings (
            user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            style TEXT,
            humor INT NOT NULL DEFAULT 5,
            caption_length TEXT,
            sensitive_mode BOOLEAN NOT NULL DEFAULT true,
            reminder_time TEXT,
            channel TEXT,
            generator TEXT NOT NULL DEFAULT 'mock',
            detail_level TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # Phase 9 tables — schema only, nothing writes to these yet.
    op.execute(
        """
        CREATE TABLE checkpoints (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            data_snapshot TEXT,
            metrics JSONB,
            artifact_url TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE training_runs (
            id BIGSERIAL PRIMARY KEY,
            kind TEXT NOT NULL,
            config JSONB,
            snapshot_hash TEXT,
            metrics JSONB,
            decision TEXT,
            reasons TEXT[] NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # Mirrors app.llm.cost.CostEvent field-for-field.
    op.execute(
        """
        CREATE TABLE cost_events (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
            day_id TEXT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            role TEXT,
            input_tokens INT NOT NULL DEFAULT 0,
            output_tokens INT NOT NULL DEFAULT 0,
            units INT NOT NULL DEFAULT 0,
            usd DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    for table in (
        "cost_events",
        "training_runs",
        "checkpoints",
        "settings",
        "thumbs",
        "caption_pairs",
        "image_pairs",
        "candidates",
        "panels",
        "beats",
        "days",
        "identity_models",
        "person_photos",
        "people",
        "jobs",
        "users",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
