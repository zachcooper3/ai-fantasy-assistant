"""
SQLite engine and session setup for the fantasy football draft assistant.
Author: Zach Cooper
"""

import os
from sqlmodel import SQLModel, Session, create_engine

# Path to the SQLite database file. Override via DB_PATH env var if needed.
_DB_PATH = os.getenv("DB_PATH", "data/fantasy.db")
_DB_URL = f"sqlite:///{_DB_PATH}"

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
