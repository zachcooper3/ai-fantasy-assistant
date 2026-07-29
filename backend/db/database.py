"""
SQLite engine and session setup for the fantasy football draft assistant.
Author: Zach Cooper
"""

import os
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine

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
    """Creates all SQLModel tables. Safe to call on every startup (no-op if already exists)."""
    SQLModel.metadata.create_all(engine)


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
