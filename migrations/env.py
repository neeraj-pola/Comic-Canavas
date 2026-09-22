import os
from logging.config import fileConfig

from alembic import context
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import engine_from_config, pool, text

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# No ORM anywhere in this codebase (raw psycopg everywhere, e.g.
# app/db.py) — every migration in versions/ is plain `op.execute(sql)`,
# never `op.create_table(...)`, so there's no metadata to autogenerate
# against.
target_metadata = None

# `DATABASE_URL` from `.env` overrides whatever placeholder is in
# alembic.ini, resolved the same way every service's own `app/config.py`
# does so `make migrate` works regardless of the caller's cwd.
load_dotenv(find_dotenv(usecwd=True))
database_url = os.environ.get("DATABASE_URL")

# `DB_SCHEMA` (tests only — real dev/prod always use the default `public`
# search path) isolates a test run's tables under their own schema so
# tests never touch dev data. Setting it here, not by hand-crafting
# `?options=-csearch_path=...` into DATABASE_URL per test, is the one
# place schema selection happens.
db_schema = os.environ.get("DB_SCHEMA")

if database_url:
    # Alembic/SQLAlchemy's engine needs the psycopg3 dialect prefix;
    # DATABASE_URL elsewhere in this project (app/config.py's Settings)
    # is a plain `postgresql://` string meant for direct psycopg use.
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if db_schema:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}options=-csearch_path%3D{db_schema},public"
    # Config.set_main_option() writes through a configparser with
    # interpolation enabled, which treats a bare `%` as starting a
    # `%(...)` reference — a DATABASE_URL with a url-encoded query string
    # breaks with "invalid interpolation syntax" unless every literal `%`
    # is doubled first.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        if db_schema:
            # search_path alone isn't enough: an unqualified
            # `alembic_version` reference resolves to whichever schema in
            # search_path the table already exists in, not the first
            # schema unconditionally, so Alembic can silently bookkeep
            # against `public` while creating zero tables in `db_schema`.
            # `version_table_schema` schema-qualifies Alembic's own version
            # table explicitly, sidestepping search_path resolution for it
            # entirely. The schema itself must exist before any DDL runs,
            # and this DDL needs its own commit — it isn't part of the
            # migration's own transaction.
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{db_schema}"'))
            connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=db_schema,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
