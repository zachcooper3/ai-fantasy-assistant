"""
Sets up a local Chroma client, creates a collection, and adds chunk dicts to the DB.

Every chunk should carry a metadata dict — at minimum `chunk_type`
("what_happened" or "what_it_means") and, where known, `sleeper_id` — so
retrieval can be scoped to one player instead of searching the whole
collection, and callers can tell factual reporting apart from AI-synthesized
analysis. See backend/ingestion/chunker.py (what_happened) and
backend/ingestion/fetch_synthesis.py (what_it_means).

Author: Zach Cooper
"""

import hashlib
import logging

import chromadb
from backend.rag.embedder import embedding_fn

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "nfl_collection"
_CHROMA_PATH = "./chroma_db"

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=embedding_fn)
    return _collection


def reset_collection() -> None:
    """
    Deletes and recreates the Chroma collection from scratch.

    add_chunks() only ever upserts what it's given — a source that used to
    produce chunks but no longer does (or produces them in a different
    shape) never gets cleaned up on its own, so old chunks just sit there
    forever, polluting every query alongside the current ones. Confirmed
    live: a collection that predated this session's chunker.py rewrite
    still had 382 chunks in the old "Player | ADP: X | Position: Y | Team:
    Z" format mixed in with the new Sleeper/RotoWire chunks, and generic
    queries were matching the stale ones more often than the real content.

    Call this once after any change to what a chunk-producing function
    outputs (a new field, a reworded template, a removed source) — routine
    re-ingestion of the *same* schema doesn't need it, since add_chunks'
    dedupe_key-based upserts already handle that correctly.
    """
    global _collection
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    try:
        client.delete_collection(name=_COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet -- nothing to reset
    _collection = None

def _chunk_id(chunk: str, meta: dict | None) -> str:
    """
    Derives a stable Chroma document ID for one chunk.

    Prefers metadata["dedupe_key"] when a caller supplies one — needed for
    "snapshot" sources where one entity has exactly one current chunk (e.g.
    a player's injury status, or their latest synthesized note) and a new
    value should REPLACE the old chunk rather than sit alongside it. Text
    alone isn't a safe key for these: two different players can render to
    identical text (confirmed live — two Sleeper entries both rendered as
    "<name> is listed as Out.", crashing the old hash-only scheme with a
    DuplicateIDError), and a status change produces different text, which
    under a text-hash ID would silently leave the old status chunk behind
    as an orphaned stale entry instead of overwriting it.

    Falls back to hashing the chunk text when no dedupe_key is given —
    correct for "event log" sources like RotoWire articles, where distinct
    events naturally have distinct text and a headline re-surfacing across
    refreshes should overwrite rather than duplicate (original behavior,
    unchanged for that case).
    """
    meta = meta or {}
    key = meta.get("dedupe_key") or chunk
    return hashlib.md5(key.encode()).hexdigest()


def add_chunks(chunks: list[str], metadatas: list[dict] | None = None) -> None:
    """
    Embeds and upserts chunks into the Chroma collection.

    IDs are derived per _chunk_id — text hash by default, or
    metadata["dedupe_key"] when a source needs one-current-chunk-per-entity
    semantics (see _chunk_id's docstring).

    metadatas, if given, must be the same length as chunks (one dict per
    chunk, may be empty but not None per-entry).

    If two input chunks land on the same ID (should only happen if a
    producer's dedupe_key isn't actually unique, or two chunks have
    identical text with no dedupe_key at all), the last one wins and a
    warning is logged — Chroma itself rejects duplicate IDs within a single
    upsert call outright, so silently deduping here is what keeps a bad
    batch from crashing the whole refresh.
    """
    if not chunks:
        return
    if metadatas is not None and len(metadatas) != len(chunks):
        raise ValueError(
            f"metadatas length ({len(metadatas)}) must match chunks length ({len(chunks)})"
        )

    metas = metadatas if metadatas is not None else [None] * len(chunks)
    by_id: dict[str, tuple[str, dict | None]] = {}
    dupes = 0
    for chunk, meta in zip(chunks, metas):
        cid = _chunk_id(chunk, meta)
        if cid in by_id:
            dupes += 1
        by_id[cid] = (chunk, meta)

    if dupes:
        logger.warning(
            f"{dupes} chunk(s) collided on the same ID within this batch — "
            "kept the last of each, dropped the rest."
        )

    ids = list(by_id.keys())
    documents = [v[0] for v in by_id.values()]
    kwargs = {"documents": documents, "ids": ids}
    if metadatas is not None:
        kwargs["metadatas"] = [v[1] for v in by_id.values()]
    get_collection().upsert(**kwargs)

def query(question: str, n_results: int = 5, where: dict | None = None) -> list[str]:
    """
    Queries the collection and returns the most relevant chunks.
    `where` is a Chroma metadata filter — e.g. {"sleeper_id": "4866"} to
    scope retrieval to one player, or {"chunk_type": "what_happened"} to
    only pull factual reporting and skip synthesized analysis.
    """
    results = get_collection().query(
        query_texts=[question],
        n_results=n_results,
        where=where,
    )
    return results["documents"][0]

def main():
    import argparse

    from backend.ingestion.chunker import build_what_happened_chunks

    parser = argparse.ArgumentParser(
        description="Ingest 'what happened' chunks and interactively query the ChromaDB collection."
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe the collection before re-ingesting. Use this once after changing "
             "what a chunk-producing function outputs, so old-format chunks don't "
             "linger and pollute query results (see reset_collection's docstring).",
    )
    args = parser.parse_args()

    if args.reset:
        reset_collection()
        print("Collection reset.")

    chunks, metadatas = build_what_happened_chunks()
    add_chunks(chunks, metadatas)
    print(f"Added {len(chunks)} chunks to the collection")
    print(
        "\nNote: this collection holds factual 'what happened' snippets (and, once "
        "you've run fetch_synthesis.py, short 'what it means' notes) — not player "
        "rankings. It's built to be queried per-player (e.g. 'Is Bijan Robinson "
        "injured?'), not as a general leaderboard ('who's the best WR?'). Ranking "
        "questions are answered by ADP/SQL in the live app, not retrieval."
    )

    # Test a query
    while True:
        user_entry = input("\nEnter a question about the NFL: ")
        if user_entry.strip() == "":
            break

        for result in query(user_entry):
            print(result)
        print()

if __name__ == "__main__":
    main()