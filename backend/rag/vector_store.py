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
import chromadb
from backend.rag.embedder import embedding_fn

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path="./chroma_db")
        _collection = client.get_or_create_collection(
            name="nfl_collection",
            embedding_function=embedding_fn)
    return _collection

def add_chunks(chunks: list[str], metadatas: list[dict] | None = None) -> None:
    """
    Embeds and upserts chunks into the Chroma collection.

    IDs are derived from the chunk text itself (unchanged from the original
    behavior), so re-ingesting the same headline or note twice overwrites
    rather than duplicates — useful since a news feed will often re-surface
    the same item across multiple refreshes.

    metadatas, if given, must be the same length as chunks (one dict per
    chunk, may be empty but not None per-entry).
    """
    if not chunks:
        return
    if metadatas is not None and len(metadatas) != len(chunks):
        raise ValueError(
            f"metadatas length ({len(metadatas)}) must match chunks length ({len(chunks)})"
        )

    ids = [hashlib.md5(chunk.encode()).hexdigest() for chunk in chunks]
    kwargs = {"documents": chunks, "ids": ids}
    if metadatas is not None:
        kwargs["metadatas"] = metadatas
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