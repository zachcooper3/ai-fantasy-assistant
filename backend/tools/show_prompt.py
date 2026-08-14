"""
Prints the exact data block an ingestion script would send to Claude for one
player — without making an API call.

Why this exists: the synthesis steps are the only part of the pipeline that
costs money, which makes them the part people are least willing to run just
to check whether a change landed. But the expensive part is the API call,
not the prompt construction — and the prompt is where the bugs live. Every
grounding bug found on this project so far (a permanently-False "Rookie or
second-year" line, red zone rates divided by the wrong denominator, an
ungrounded DST note) was visible in the rendered prompt before any token was
spent.

    py -m backend.tools.show_prompt "Jahmyr Gibbs"
    py -m backend.tools.show_prompt "Gibbs" --rookie   # DraftProfile version
    py -m backend.tools.show_prompt --sleeper-id 4866

Reads whatever database DB_PATH points at, so it's safe to aim at a copy:

    set DB_PATH=C:\\tmp\\verify.db
    py -m backend.tools.show_prompt "Bijan"

Author: Zach Cooper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlmodel import Session, select

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import engine
from backend.db.models import DraftProfile, Player, PlayerMetrics


def _veteran(session: Session, name: str | None, sleeper_id: str | None):
    """The fetch_synthesis.py path — grounded in PlayerMetrics."""
    from backend.ingestion.fetch_synthesis import format_metrics_prompt

    # Joined on sleeper_id rather than player_id, matching what
    # fetch_synthesis.py itself does — see metrics_repo.py's docstring.
    query = select(Player, PlayerMetrics).where(
        PlayerMetrics.sleeper_id == Player.sleeper_id
    )
    if sleeper_id:
        query = query.where(Player.sleeper_id == sleeper_id)
    if name:
        query = query.where(Player.name.ilike(f"%{name}%"))  # type: ignore[attr-defined]

    row = session.exec(query).first()
    if row is None:
        return None
    return format_metrics_prompt(*row)


def _rookie(session: Session, name: str | None, sleeper_id: str | None):
    """The fetch_rookie_synthesis.py path — grounded in DraftProfile."""
    from backend.ingestion.fetch_rookie_synthesis import format_draft_profile_prompt

    query = select(Player, DraftProfile).where(
        DraftProfile.sleeper_id == Player.sleeper_id
    )
    if sleeper_id:
        query = query.where(Player.sleeper_id == sleeper_id)
    if name:
        query = query.where(Player.name.ilike(f"%{name}%"))  # type: ignore[attr-defined]

    row = session.exec(query).first()
    if row is None:
        return None
    return format_draft_profile_prompt(*row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the Claude synthesis prompt for one player without calling the API."
    )
    parser.add_argument("name", nargs="?", default=None,
                        help="Substring of the player's name (case-insensitive).")
    parser.add_argument("--sleeper-id", default=None, help="Exact sleeper_id instead of a name.")
    parser.add_argument("--rookie", action="store_true",
                        help="Render the DraftProfile-grounded rookie prompt instead of the "
                             "PlayerMetrics-grounded veteran one.")
    args = parser.parse_args()

    if not args.name and not args.sleeper_id:
        parser.error("give a name or --sleeper-id")

    with Session(engine) as session:
        render = _rookie if args.rookie else _veteran
        prompt = render(session, args.name, args.sleeper_id)

    if prompt is None:
        which = "DraftProfile" if args.rookie else "PlayerMetrics"
        print(
            f"No player matched with a {which} row. Either the name doesn't "
            f"match, or that player has no {which} row — which is itself the "
            f"answer: the synthesis step would skip them.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(prompt)


if __name__ == "__main__":
    main()
