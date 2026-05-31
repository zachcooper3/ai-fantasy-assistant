"""
Chunker functions for ingested data
Author: Zach Cooper
"""

import csv

def chunk_csv_row(row: dict[str, str]) -> str:
    """
    Creates and returns a text chunk from a CSV row for ChromaDB ingestion.
    This text is embedded and used to provide semantic context to Claude.
    Structured queries (filtering, sorting) use SQLite instead — see backend/db/.
    """
    bye = row.get("Bye", "").strip() or "TBD"
    return (
        f"{row.get('Player', 'Unknown')} | Rank: {row.get('Rank', 'N/A')} "
        f"| ADP: {row.get('AVG', 'N/A')} "
        f"| Position: {row.get('POS', 'N/A')} "
        f"| Team: {row.get('Team', 'N/A')} "
        f"| Bye: {bye}"
    )

def chunk_csv_file(filename: str) -> list[str]:
    """Creates and returns a list of chunks from a CSV file"""
    chunks = []
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            chunks.append(chunk_csv_row(row))

    return chunks

def main():
    # Manual tests for chunking
    chunks = chunk_csv_file("data/raw/fantasypros_adp.csv")

    for chunk in chunks:
        print(chunk)

if __name__ == "__main__":
    main()