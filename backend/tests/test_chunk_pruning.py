"""
Tests for snapshot-chunk reconciliation (chunker.prune_stale_snapshots /
vector_store.prune_source) and for the retrieval ranking that decides which
chunks survive _MAX_CHUNKS_PER_PLAYER.

Together these cover the "the store says a healthy player is injured" bug:
one half stops writing it, the other stops it from crowding out the chunks
that matter when it does exist.

Author: Zach Cooper
"""

import sys
import types

import pytest

from backend.app.services.ai_service import _MAX_CHUNKS_PER_PLAYER, _rank_chunks
from backend.ingestion import chunker


# ---------------------------------------------------------------------------
# chunk_sleeper_injuries — what goes in
# ---------------------------------------------------------------------------

def test_healthy_players_produce_no_chunk():
    players = {
        "1": {"full_name": "Healthy Guy", "injury_status": ""},
        "2": {"full_name": "Also Fine", "injury_status": None},
    }
    chunks, metas = chunker.chunk_sleeper_injuries(players)
    assert chunks == [] and metas == []


def test_injured_player_chunk_carries_a_stable_dedupe_key():
    players = {"77": {"full_name": "Sore Back", "injury_status": "Questionable",
                      "injury_body_part": "Hamstring"}}
    chunks, metas = chunker.chunk_sleeper_injuries(players)
    assert "Questionable" in chunks[0] and "Hamstring" in chunks[0]
    assert metas[0]["dedupe_key"] == "sleeper_injury:77"
    assert metas[0]["source"] == "sleeper_injury_status"


# ---------------------------------------------------------------------------
# prune_stale_snapshots — what comes out
# ---------------------------------------------------------------------------

class _FakeStore:
    """Stands in for vector_store.prune_source, recording what it was asked
    to keep. The real one talks to Chroma; the contract worth pinning here
    is which keys get passed and whether it's called at all."""

    def __init__(self):
        self.calls: list[tuple[str, set[str]]] = []

    def __call__(self, source: str, keep: set[str]) -> int:
        self.calls.append((source, keep))
        return 0


@pytest.fixture
def fake_prune(monkeypatch):
    """Substitutes the whole vector_store module rather than patching an
    attribute on it. Importing the real one pulls in chromadb, which is a
    heavy optional dependency the rest of this suite deliberately runs
    without — and prune_stale_snapshots imports it lazily inside the
    function body, so there's no already-imported attribute to patch."""
    store = _FakeStore()
    stub = types.ModuleType("backend.rag.vector_store")
    stub.prune_source = store
    monkeypatch.setitem(sys.modules, "backend.rag.vector_store", stub)
    return store


def test_prune_keeps_exactly_the_players_still_injured(fake_prune):
    metas = [
        {"source": "sleeper_injury_status", "dedupe_key": "sleeper_injury:1"},
        {"source": "sleeper_injury_status", "dedupe_key": "sleeper_injury:2"},
    ]
    chunker.prune_stale_snapshots(metas, {"sleeper_injury_status"})

    source, keep = fake_prune.calls[0]
    assert source == "sleeper_injury_status"
    assert keep == {"sleeper_injury:1", "sleeper_injury:2"}


def test_prune_is_skipped_when_the_sleeper_fetch_failed(fake_prune):
    """The whole point of the prunable_sources set. An empty chunk list from
    a failed fetch must not be read as 'nobody is injured any more'."""
    chunker.prune_stale_snapshots([], prunable_sources=set())
    assert fake_prune.calls == []


def test_prune_runs_with_an_empty_keep_set_when_the_fetch_succeeded(fake_prune):
    """The legitimate empty case: Sleeper answered, and genuinely nobody
    carries a designation. Every existing chunk is now stale."""
    chunker.prune_stale_snapshots([], {"sleeper_injury_status"})
    assert fake_prune.calls == [("sleeper_injury_status", set())]


def test_prune_never_touches_the_news_feed(fake_prune):
    """RotoWire is an event log, not a snapshot — the feed only carries a
    rolling ~40 items, so pruning to it would delete every older story."""
    metas = [{"source": "rotowire_rss", "link": "http://x"}]
    chunker.prune_stale_snapshots(metas, {"rotowire_rss", "sleeper_injury_status"})

    pruned_sources = [source for source, _ in fake_prune.calls]
    assert "rotowire_rss" not in pruned_sources


# ---------------------------------------------------------------------------
# _rank_chunks — which survive the cap
# ---------------------------------------------------------------------------

def _chunk(source: str, doc: str, when: str = ""):
    meta = {"source": source}
    if when:
        meta["pub_date" if source == "rotowire_rss" else "generated_at"] = when
    return (meta, doc)


def test_injury_status_survives_the_cap_over_news():
    entries = [
        _chunk("rotowire_rss", "news 1"),
        _chunk("rotowire_rss", "news 2"),
        _chunk("rotowire_rss", "news 3"),
        _chunk("sleeper_injury_status", "IR"),
    ]
    kept = _rank_chunks(entries)[:_MAX_CHUNKS_PER_PLAYER]
    assert "IR" in kept


def test_synthesis_survives_the_cap_over_news():
    entries = [
        _chunk("rotowire_rss", "news 1"),
        _chunk("rotowire_rss", "news 2"),
        _chunk("rotowire_rss", "news 3"),
        _chunk("claude_synthesis", "the analysis"),
    ]
    kept = _rank_chunks(entries)[:_MAX_CHUNKS_PER_PLAYER]
    assert "the analysis" in kept


def test_full_ordering_is_injury_then_synthesis_then_news():
    entries = [
        _chunk("rotowire_rss", "news"),
        _chunk("claude_synthesis", "analysis"),
        _chunk("sleeper_injury_status", "injury"),
    ]
    assert _rank_chunks(entries) == ["injury", "analysis", "news"]


def test_rookie_synthesis_ranks_alongside_veteran_synthesis():
    entries = [
        _chunk("rotowire_rss", "news"),
        _chunk("claude_synthesis_rookie", "rookie analysis"),
    ]
    assert _rank_chunks(entries) == ["rookie analysis", "news"]


def test_newer_news_wins_within_the_same_tier():
    entries = [
        _chunk("rotowire_rss", "old", when="2026-08-01"),
        _chunk("rotowire_rss", "new", when="2026-08-12"),
    ]
    assert _rank_chunks(entries) == ["new", "old"]


def test_unknown_source_sorts_last_but_is_not_dropped():
    entries = [
        _chunk("something_new", "mystery"),
        _chunk("sleeper_injury_status", "injury"),
    ]
    assert _rank_chunks(entries) == ["injury", "mystery"]


def test_ranking_tolerates_missing_metadata():
    assert _rank_chunks([({}, "bare"), (None, "none")]) == ["bare", "none"]


def test_ranking_of_an_empty_list_is_empty():
    assert _rank_chunks([]) == []
