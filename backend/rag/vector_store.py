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

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path="./chroma_db")
        _collection = client.get_or_create_collection(
            name="nfl_collection",
            embedding_function=embedding_fn)
    return _collection

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
    from backend.ingestion.chunker import build_what_happened_chunks

    chunks, metadatas = build_what_happened_chunks()
    add_chunks(chunks, metadatas)
    print(f"Added {len(chunks)} chunks to the collection")

    # Test a query
    while True:
        user_entry = input("Enter a question about the NFL: ")
        if user_entry.strip() == "":
            break

        for result in query(user_entry):
            print(result)
        print()

if __name__ == "__main__":
    main()