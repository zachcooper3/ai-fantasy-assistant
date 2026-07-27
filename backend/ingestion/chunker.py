"""
Builds "what happened" text chunks for ChromaDB — real, factual event
reporting, as distinct from the "what it means" analytical layer.

Two chunk types feed the vector store (see backend/app/services/ai_service.py
for how they're retrieved and used):

  "what happened"  (this file) — factual reporting only: injury status
    changes and breaking news, sourced from Sleeper's injury_status field
    and RotoWire's NFL news RSS feed (both explicitly free/licensed for this
    kind of reuse — see README's "Player Analytics & News" section).

  "what it means"   (backend/ingestion/fetch_synthesis.py) — Claude-
    generated analysis derived from computed PlayerMetrics, not scraped
    from anywhere.

Previously this module turned ADP CSV rows into text chunks. That was
always meant as a placeholder — ADP is structured, arithmetic data that
belongs in SQLite, not a vector store (see system-design.md's original
"Dual-Store Architecture" rule of thumb). Replaced entirely now that real
news ingestion exists; if you're looking for the old chunk_csv_row /
chunk_csv_file functions, they're gone on purpose, not missing by accident.

Run manually:
    py -m backend.ingestion.chunker

Author: Zach Cooper
"""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET

import httpx
from sqlmodel import Session, select

from backend.db.database import engine
from backend.db.models import Player

logger = logging.getLogger(__name__)

ROTOWIRE_NFL_RSS_URL = "https://www.rotowire.com/rss/news.php?sport=NFL"

# Sleeper injury_status values worth chunking. "" (healthy) isn't
# newsworthy and would just add noise to every healthy player's retrieval.
_NOTABLE_INJURY_STATUSES = {
    "questionable", "doubtful", "out", "ir", "pup", "suspended", "dnr", "nfi",
}


# ---------------------------------------------------------------------------
# Name matching — best-effort, never blocks chunk creation on a miss
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lowercase, strip punctuation/whitespace — same idea as
    sync_sleeper_ids.py's normalizer, duplicated here rather than imported
    since it's three lines and not worth an inter-module coupling for."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _extract_player_name(title: str) -> str | None:
    """RotoWire NFL news titles are conventionally '{Player Name}: {headline}'."""
    if ":" not in title:
        return None
    name = title.split(":", 1)[0].strip()
    return name or None


def _match_player(session: Session, name: str) -> Player | None:
    """Best-effort name match against local Player rows. Returns None on
    no match or an ambiguous one — a wrong player tag is worse than no tag,
    since it would misfile a chunk under someone else's retrieval scope."""
    normalised = _normalise(name)
    candidates = session.exec(select(Player)).all()
    matches = [p for p in candidates if _normalise(p.name) == normalised]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# RotoWire RSS
# ---------------------------------------------------------------------------

def fetch_rotowire_news(url: str = ROTOWIRE_NFL_RSS_URL, timeout: float = 15.0) -> list[dict]:
    """
    Fetches and parses RotoWire's NFL news RSS feed into plain dicts:
    {title, description, link, pub_date}. Standard RSS 2.0 structure
    (channel > item > title/description/link/pubDate) — see
    https://www.rotowire.com/rss/ (explicitly published for this kind of
    reuse on other sites).
    """
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    items = []
    for item in root.findall("./channel/item"):
        items.append({
            "title": (item.findtext("title") or "").strip(),
            "description": (item.findtext("description") or "").strip(),
            "link": (item.findtext("link") or "").strip(),
            "pub_date": (item.findtext("pubDate") or "").strip(),
        })
    return items


def chunk_rotowire_news(items: list[dict], session: Session | None = None) -> tuple[list[str], list[dict]]:
    """
    Converts RotoWire news items into (chunk_text, metadata) pairs.
    If a DB session is given, attempts to tag each chunk with the matching
    local player (sleeper_id) by parsing the "{Player}: {headline}" title
    convention — best-effort; an unmatched or ambiguous name still produces
    a chunk, just without a player-scoped tag.
    """
    chunks, metadatas = [], []
    for item in items:
        if not item["title"]:
            continue

        text = f"{item['title']} — {item['description']}".strip(" —")
        meta = {
            "chunk_type": "what_happened",
            "source": "rotowire_rss",
            "link": item["link"],
            "pub_date": item["pub_date"],
        }

        if session is not None:
            name = _extract_player_name(item["title"])
            if name:
                player = _match_player(session, name)
                if player is not None:
                    meta["player_name"] = player.name
                    if player.sleeper_id:
                        meta["sleeper_id"] = player.sleeper_id

        chunks.append(text)
        metadatas.append(meta)

    return chunks, metadatas


# ---------------------------------------------------------------------------
# Sleeper injury_status
# ---------------------------------------------------------------------------

def chunk_sleeper_injuries(sleeper_players: dict[str, dict]) -> tuple[list[str], list[dict]]:
    """
    Converts Sleeper's injury_status field into chunk text for every player
    with a notable (non-empty, non-healthy) designation. sleeper_players is
    the raw {sleeper_id: player_dict} payload from
    backend/app/services/sleeper_client.get_nfl_players() — the same data
    sync_sleeper_ids.py already fetches, just read here for a different
    field (injury_status/injury_body_part instead of full_name/position).
    """
    chunks, metadatas = [], []
    for sleeper_id, p in sleeper_players.items():
        status = (p.get("injury_status") or "").strip()
        if status.lower() not in _NOTABLE_INJURY_STATUSES:
            continue

        name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        if not name:
            continue

        body_part = p.get("injury_body_part")
        detail = f" ({body_part})" if body_part else ""
        text = f"{name} is listed as {status}{detail}."

        chunks.append(text)
        metadatas.append({
            "chunk_type": "what_happened",
            "source": "sleeper_injury_status",
            "player_name": name,
            "sleeper_id": sleeper_id,
            # One current status per player — a later status should
            # replace this chunk, not accumulate alongside it, and two
            # different players can render identical text (e.g. same name,
            # same status, no body part), so text alone can't be the key.
            # See vector_store._chunk_id.
            "dedupe_key": f"sleeper_injury:{sleeper_id}",
        })

    return chunks, metadatas


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_what_happened_chunks() -> tuple[list[str], list[dict]]:
    """
    Pulls both sources and returns combined (chunks, metadatas) ready for
    backend/rag/vector_store.add_chunks(). Each source is isolated — a
    RotoWire outage shouldn't prevent Sleeper injury chunks (or vice versa)
    from still being ingested, same "one flaky source shouldn't take down
    the whole refresh" principle as fetch_metrics.py.
    """
    all_chunks: list[str] = []
    all_metadatas: list[dict] = []

    try:
        news_items = fetch_rotowire_news()
    except Exception as e:
        news_items = None
        logger.warning(f"Could not fetch RotoWire RSS — news chunks skipped: {e}")

    if news_items is not None:
        try:
            with Session(engine) as session:
                news_chunks, news_meta = chunk_rotowire_news(news_items, session=session)
            all_chunks += news_chunks
            all_metadatas += news_meta
            logger.info(f"RotoWire: {len(news_chunks)} news chunks")
        except Exception as e:
            logger.warning(f"Fetched RotoWire RSS but could not chunk it (player-matching DB lookup failed) — news chunks skipped: {e}")

    try:
        from backend.app.services.sleeper_client import get_nfl_players
        sleeper_players = asyncio.run(get_nfl_players())
        injury_chunks, injury_meta = chunk_sleeper_injuries(sleeper_players)
        all_chunks += injury_chunks
        all_metadatas += injury_meta
        logger.info(f"Sleeper: {len(injury_chunks)} injury-status chunks")
    except Exception as e:
        logger.warning(f"Could not fetch Sleeper player data — injury chunks skipped: {e}")

    return all_chunks, all_metadatas


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from backend.rag.vector_store import add_chunks

    chunks, metadatas = build_what_happened_chunks()
    add_chunks(chunks, metadatas)
    print(f"Added {len(chunks)} 'what happened' chunks to the collection.")


if __name__ == "__main__":
    main()
