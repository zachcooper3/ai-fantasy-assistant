"""
Adds columns to tables that already exist.

SQLModel's create_all() only ever CREATEs — it is a no-op against a table
that is already there, whatever the model now says. So adding a field to a
model does nothing to a database created before it, and the first query
touching that field fails with "no such column". On this app that means a
draft-day 500 on a schema change nobody thought was risky.

Deliberately not Alembic. This needs to add nullable columns to a
single-user SQLite file, and a migrations framework brings a version table,
a revision history and a CLI step before every run — all of which have to be
right on draft day to gain nothing over one ALTER. If this ever needs to
rename or backfill a column, that trade flips and Alembic is the answer.

Every migration here must be:
  - idempotent — it runs on every startup
  - additive only — nullable columns, no drops, no type changes
  - safe on an empty database, where create_all has already made the column

Author: Zach Cooper
"""

import logging

from sqlalchemy import inspect, text
from sqlmodel import Session

logger = logging.getLogger(__name__)

# (table, column, SQLite type) — added when absent, in this order.
#
# Both of these exist to give the recommendation prompt forward-looking
# context, which is otherwise its weakest point: everything else it knows
# about a player describes a season that has already finished.
_COLUMNS: list[tuple[str, str, str]] = [
    # Where a player earned last season's numbers. Without it, a target
    # share is just a number — with it, the app can see that the player who
    # took 25% of a team's targets is now somewhere else, and that whoever
    # stayed is competing for a different amount of work than the raw share
    # implies. See _format_roster_changes in ai_service.py.
    ("playermetrics", "team", "VARCHAR"),
    # Sleeper's injury designation (IR, Out, PUP, suspended, questionable).
    # Currently this only reaches the prompt as prose inside a retrieved
    # ChromaDB chunk, which has to be fetched AND correctly interpreted —
    # and confirmed live, a player listed IR was recommended anyway with the
    # note sitting in the prompt. A column can be rendered on the board and
    # checked in Python.
    ("player", "injury_status", "VARCHAR"),
    # The AI panel's Haiku/Sonnet toggle (see AI_MODEL_CHOICES in
    # ai_service.py) — persisted so a mid-draft backend restart resumes on
    # whichever model you'd switched to instead of silently reverting to
    # the CLAUDE_MODEL env default.
    ("draftsession", "ai_model", "VARCHAR"),
]


def _existing_columns(session: Session, table: str) -> set[str]:
    inspector = inspect(session.get_bind())
    if table not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def run_migrations(engine) -> list[str]:
    """
    Applies any missing column additions. Returns what it added, so startup
    can log it — a schema change that happens silently is one nobody
    notices went wrong.

    Never raises: a failure here must not stop the server from booting. A
    missing column degrades one prompt section; a backend that won't start
    ends a draft.
    """
    applied: list[str] = []
    try:
        with Session(engine) as session:
            for table, column, sql_type in _COLUMNS:
                present = _existing_columns(session, table)
                if not present:
                    # Table doesn't exist yet — create_all is about to make
                    # it, with this column already in the model.
                    continue
                if column in present:
                    continue
                session.exec(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))
                session.commit()
                applied.append(f"{table}.{column}")
                logger.info("Migration: added %s.%s", table, column)
    except Exception:
        logger.exception(
            "Column migration failed — the app will still start, but any prompt "
            "section depending on the new column will be silently absent."
        )
    return applied
