"""
Sets up a local Chroma client, creates a collection, and adds chunk dicts to the DB
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

def add_chunks(chunks: list[str]) -> None:
    """Embeds and adds a list of chunk strings to the Chroma collection"""
    ids = [hashlib.md5(chunk.encode()).hexdigest() for chunk in chunks]
    get_collection().upsert(
        documents=chunks,
        ids=ids
    )

def query(question: str, n_results: int = 5) -> list[str]:
    """Queries the collection and returns the most relevant chunks"""
    results = get_collection().query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0]

def main():
    from backend.ingestion.chunker import chunk_csv_file

    # Load and store chunks
    chunks = chunk_csv_file("data/raw/fantasypros_adp.csv")
    add_chunks(chunks)
    print(f"Added {len(chunks)} chunks to the collection")

    # Test a query
    while True:
        # Gather user input
        user_entry = input("Enter a question about the NFL: ")
        lowercase_entry = user_entry.lower()

        if (lowercase_entry == ""):
            break

        # Send query
        results = query(user_entry)

        # Print retrieval results
        for result in results:
            print(result)
        print()

if __name__ == "__main__":
    main()