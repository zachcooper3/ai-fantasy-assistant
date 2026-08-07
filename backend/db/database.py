"""
SQLite engine and session setup for the fantasy football draft assistant.
Author: Zach Cooper
"""

import logging
import os
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

# Repo root — backend/db/database.py is two levels below it.
logger = logging.getLogger(__name__)

# Repo root — backend/db/database.py is two levels below it.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Path to the SQLite database file. Override via DB_PATH env var if needed.
# A relative path (including the default) is resolved against the REPO
# ROOT, not the process CWD — launching uvicorn from any other directory
# used to silently create a fresh empty DB wherever you happened to be
# standing (audit W10). Absolute overrides are used as-is.
_DB_PATH = Path(os.getenv("DB_PATH", "data/fantasy.db"))
if not _DB_PATH.is_absolute():
    _DB_PATH = _REPO_ROOT / _DB_PATH
_DB_URL = f"sqlite:///{_DB_PATH.as_posix()}"

# connect_args required for SQLite to work with FastAPI's threaded request handling
engine = create_engine(
    _DB_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # Set to True to log all SQL queries during development
)


def create_db_and_tables() -> None:
    """
    Creates all SQLModel tables, then applies any additive column
    migrations. Safe to call on every startup.

    Order matters: create_all first so a brand-new database gets complete
    tables and the migrations find nothing to do, then run_migrations to
    catch databases created before a field was added. create_all alone
    never alters an existing table, which is how a model change can leave
    a live database missing a column it now assumes.
    """
    SQLModel.metadata.create_all(engine)
    from backend.db.migrations import run_migrations

    applied = run_migrations(engine)
    if applied:
        logger.info("Applied column migrations: %s", ", ".join(applied))


def get_session():
    """
    FastAPI dependency that yields a database session per request.

    Usage in a route:
        @app.get("/players")
        def list_players(session: Session = Depends(get_session)):
            ...
    """
    with Session(engine) as session:
        yield session
