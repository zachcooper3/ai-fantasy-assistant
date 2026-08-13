"""
Enriches existing DraftProfile rows with final-college-season production
(passing/rushing/receiving), from CollegeFootballData.com.

Only enriches players who already have a DraftProfile row — i.e., actually
appear in a recent NFL draft class (see fetch_draft_profiles.py, which must
run first). An undrafted rookie with real college production is a real but
rare gap this doesn't cover; adding it would mean giving every DraftProfile
row a nullable draft_year instead of a required one, which felt like the
wrong tradeoff for how rarely it matters. Revisit if it turns out to matter.

Run manually (after fetch_draft_profiles.py):
    py -m backend.ingestion.fetch_college_stats                  # all draft classes on file
    py -m backend.ingestion.fetch_college_stats --dry-run         # print matches, don't write
    py -m backend.ingestion.fetch_college_stats --sleeper-id 1234 # debug one player

Requires CFBD_API_KEY (see .env.example) — free, no credit card, email
signup at https://collegefootballdata.com/key. No-ops with a clear log
message if it isn't set, same "draft day shouldn't crash over a missing
optional feature" stance as the rest of this app's ingestion scripts.

A note on verification: written without a CFBD_API_KEY available in the
dev sandbox to test against, so — same disclaimer as fetch_metrics.py and
fetch_draft_profiles.py carry for their own third-party data sources —
the exact response key casing (statType vs stat_type, etc.) and the exact
stat_type label strings (e.g. is it "YDS" or "yds" or "yards"?) are
resolved defensively against a few plausible candidates, not assumed
correct. Run with --dry-run first and check the logged per-player output;
if college production comes back empty for everyone, that's the thing to
fix (log the raw row shape and adjust the candidate lists below), not a
sign the whole approach is broken.

Author: Zach Cooper
"""

import argparse
import logging
import sys
from pathlib import Path

from sqlmodel import Session, select

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.services import cfbd_client
from backend.db import draft_profile_repo
from backend.db.database import engine
from backend.db.models import DraftProfile, Player
from backend.ingestion.fetch_draft_profiles import _normalise, _strip_suffix
from backend.ingestion.fetch_metrics import _first

logger = logging.getLogger(__name__)

# Pulled per season, then pivoted per player. Position isn't in
# PlayerSeasonStat's schema (confirmed from CFBD's own API docs), so a
# rushing-heavy QB's rushing line and a receiving RB's receiving line both
# just show up naturally — no need to filter by category per position.
_CATEGORIES = ("passing", "rushing", "receiving")

# (DraftProfile field, category, stat_type candidates). stat_type string
# values aren't confirmed live (see module docstring) — a few plausible
# spellings per stat, tried in order.
_STAT_FIELDS: list[tuple[str, str, tuple[str, ...]]] = [
    ("passing_yards", "passing", ("YDS", "yds", "yards")),
    ("passing_td", "passing", ("TD", "td", "touchdowns")),
    ("interceptions_thrown", "passing", ("INT", "int", "interceptions")),
    ("rushing_yards", "rushing", ("YDS", "yds", "yards")),
    ("rushing_td", "rushing", ("TD", "td", "touchdowns")),
    ("carries", "rushing", ("CAR", "car", "attempts", "ATT")),
    ("receiving_yards", "receiving", ("YDS", "yds", "yards")),
    ("receiving_td", "receiving", ("TD", "td", "touchdowns")),
    ("receptions", "receiving", ("REC", "rec", "receptions")),
]

# Common first-name nickname/formal pairs. Added after a live 2026-07-29
# run where "Cam Ward", "Cam Skattebo", "Woody Marks", and "Mike
# Washington Jr." all failed to match despite being real, well-documented
# college seasons — nflreadpy's draft_picks matched all of these against
# our own Player.name just fine (see fetch_draft_profiles.py), so the
# nickname/formal split is specifically a CFBD-vs-everyone-else thing, not
# an error on our side. NOT verified against a raw CFBD row (network-
# restricted dev sandbox — see module docstring): this is a best-effort
# fallback to retry with, not a confirmed mapping. If match counts don't
# improve after this, the real explanation is something else and this
# table isn't it.
_NICKNAME_TO_FORMAL = {
    "cam": "cameron", "mike": "michael", "woody": "woodrow",
    "zach": "zachary", "alex": "alexander", "chris": "christopher",
    "nick": "nicholas", "matt": "matthew", "will": "william",
    "sam": "samuel", "tony": "anthony", "rob": "robert",
    "dan": "daniel", "joe": "joseph", "ben": "benjamin",
    "nate": "nathaniel", "josh": "joshua", "jake": "jacob",
}
_FORMAL_TO_NICKNAME = {formal: nick for nick, formal in _NICKNAME_TO_FORMAL.items()}


# ---------------------------------------------------------------------------
# Fetch + pivot
# ---------------------------------------------------------------------------

def _fetch_season(season: int) -> dict[str, dict[str, dict[str, float]]]:
    """
    Pulls all three stat categories for one college season and pivots
    CFBD's long/narrow rows into {normalised_player_name: {category:
    {stat_type_lower: value}}}.

    Stores each player under BOTH their raw normalised name and a
    suffix-stripped variant (when they have one) — same fix as
    fetch_draft_profiles.py's _build_player_index, applied here for the
    same reason: confirmed via live nflreadpy data that generational
    suffixes ("Jr.", "III") aren't reliably present on only one side of a
    name match. Whichever side (CFBD's or our own Player.name) carries the
    suffix, widening this index to hold both forms covers it regardless of
    which direction the mismatch runs.
    """
    by_player: dict[str, dict[str, dict[str, float]]] = {}

    for category in _CATEGORIES:
        try:
            rows = cfbd_client.get_player_season_stats(year=season, category=category)
        except Exception as e:
            logger.warning(f"Could not fetch {category} stats for {season}: {e}")
            continue

        for row in rows:
            name = _first(row, "player", "playerName", "name")
            stat_type = _first(row, "statType", "stat_type", "category_type")
            stat = _first(row, "stat")
            if not name or stat_type is None or stat is None:
                continue

            key = _normalise(name)
            entry = by_player.setdefault(key, {}).setdefault(category, {})
            entry[str(stat_type).lower()] = stat

            stripped = _strip_suffix(key)
            if stripped is not None:
                by_player.setdefault(stripped, {}).setdefault(category, {}).update(
                    {str(stat_type).lower(): stat}
                )

        logger.info(f"{season} {category}: {len(rows):,} raw rows")

    return by_player


def _name_candidates(name: str) -> list[str]:
    """All normalised forms worth trying for one player name against
    by_player: the name as given, its suffix-stripped form, and — for
    either — a first-name nickname/formal swap (see _NICKNAME_TO_FORMAL).
    First hit wins at the call site; order here doesn't imply preference."""
    bases = [_normalise(name)]
    stripped = _strip_suffix(bases[0])
    if stripped is not None:
        bases.append(stripped)

    candidates = list(bases)
    for base in bases:
        parts = base.split()
        if not parts:
            continue
        first = parts[0]
        swapped = _NICKNAME_TO_FORMAL.get(first) or _FORMAL_TO_NICKNAME.get(first)
        if swapped:
            candidates.append(" ".join([swapped, *parts[1:]]))
    return candidates


def _lookup_player_stats(
    by_player: dict[str, dict[str, dict[str, float]]], player_name: str
) -> dict[str, dict[str, float]] | None:
    """Tries every name candidate (see _name_candidates) against one
    season's pivoted CFBD data, returning the first hit."""
    for candidate in _name_candidates(player_name):
        if candidate in by_player:
            return by_player[candidate]
    return None


def _extract_fields(player_stats: dict[str, dict[str, float]]) -> dict[str, int]:
    """Pulls the named DraftProfile fields out of one player's pivoted
    category/stat_type data, trying each candidate spelling in turn."""
    fields: dict[str, int] = {}
    for field_name, category, candidates in _STAT_FIELDS:
        cat_stats = player_stats.get(category, {})
        for candidate in candidates:
            if candidate.lower() in cat_stats:
                try:
                    fields[field_name] = int(cat_stats[candidate.lower()])
                except (TypeError, ValueError):
                    pass
                break
    return fields


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def refresh_college_stats(sleeper_id: str | None = None) -> tuple[int, int]:
    """
    Enriches every existing DraftProfile row (or just one, via sleeper_id)
    with college production for their final college season. Returns
    (matched, unmatched).
    """
    if not cfbd_client.is_configured():
        logger.warning("CFBD_API_KEY not set — skipping college stats enrichment.")
        return 0, 0

    with Session(engine) as session:
        # Joined on sleeper_id, not the player_id FK. player_id is reassigned
        # wholesale on every ADP reingest (see metrics_repo.py's docstring for
        # the Gibbs/Robinson incident), so a join on it can pair a draft
        # profile with a different real player and write one prospect's
        # college production onto another's row. This was the last module
        # still using player_id for a cross-table join; everything else —
        # fetch_synthesis, fetch_rookie_synthesis, both relink helpers —
        # already keys off sleeper_id for exactly this reason.
        query = select(DraftProfile, Player).where(
            DraftProfile.sleeper_id == Player.sleeper_id
        )
        if sleeper_id:
            query = query.where(DraftProfile.sleeper_id == sleeper_id)
        rows = session.exec(query).all()

        if not rows:
            logger.info("No DraftProfile rows to enrich — run fetch_draft_profiles.py first.")
            return 0, 0

        # One API pull per distinct college season needed, not per player.
        needed_seasons = {dp.draft_year - 1 for dp, _ in rows}
        season_data = {season: _fetch_season(season) for season in needed_seasons}

        matched = unmatched = 0
        for draft_profile, player in rows:
            season = draft_profile.draft_year - 1
            by_player = season_data.get(season, {})

            player_stats = _lookup_player_stats(by_player, player.name)

            if player_stats is None:
                unmatched += 1
                continue

            fields = _extract_fields(player_stats)
            if not fields:
                unmatched += 1
                continue

            draft_profile_repo.upsert_draft_profile(
                session,
                player_id=player.id,
                sleeper_id=player.sleeper_id,
                college_season=season,
                **fields,
            )
            matched += 1

    logger.info(f"College stats: {matched} matched, {unmatched} unmatched")
    return matched, unmatched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Enrich DraftProfile rows with final-college-season production stats.")
    parser.add_argument("--sleeper-id", type=str, default=None, help="Only process this one player (debugging).")
    parser.add_argument("--dry-run", action="store_true", help="Log matches without writing to the database.")
    args = parser.parse_args()

    if not cfbd_client.is_configured():
        print(
            "CFBD_API_KEY is not set (see .env.example — free key at "
            "https://collegefootballdata.com/key). Nothing to do."
        )
        return

    if args.dry_run:
        with Session(engine) as session:
            # Same sleeper_id join as refresh_college_stats — see its comment.
            query = select(DraftProfile, Player).where(
                DraftProfile.sleeper_id == Player.sleeper_id
            )
            if args.sleeper_id:
                query = query.where(DraftProfile.sleeper_id == args.sleeper_id)
            rows = session.exec(query).all()

            if not rows:
                print(
                    "No DraftProfile rows found — run fetch_draft_profiles.py "
                    "(without --dry-run) first so there's something to enrich."
                )
                return

            needed_seasons = {dp.draft_year - 1 for dp, _ in rows}
            season_data = {season: _fetch_season(season) for season in needed_seasons}

            for draft_profile, player in rows:
                season = draft_profile.draft_year - 1
                by_player = season_data.get(season, {})
                player_stats = _lookup_player_stats(by_player, player.name)
                fields = _extract_fields(player_stats) if player_stats else {}
                status = fields if fields else "NO MATCH / NO USABLE STATS"
                print(f"{player.name} ({player.position}, college season {season}) -> {status}")
        return

    matched, unmatched = refresh_college_stats(sleeper_id=args.sleeper_id)
    print(f"Done. {matched} matched, {unmatched} unmatched.")


if __name__ == "__main__":
    main()
