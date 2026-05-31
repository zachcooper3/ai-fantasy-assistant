"""
Uses Chroma's embedding function to convert chunk text into a vector
Author: Zach Cooper
"""

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

embedding_fn = DefaultEmbeddingFunction()

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Converts a list of text chunks into a list of embedding vectors"""
    return embedding_fn(chunks)

def embed_chunk(chunk: str) -> list[float]:
    """Converts a single text chunk into a list of embedding vectors"""
    return embedding_fn([chunk])[0]

def main():
    # Manual tests for embedder
    v1 = embed_chunk("Drake Maye | ADP: 1.2 | Position: QB | Team: NE")
    print(len(v1))
    print(type(v1[0]))

    v2 = embed_chunk("Patrick Mahomes | ADP: 1.2 | Position: QB | Team: KC")
    print(len(v2))
    print(type(v2[0]))

if __name__ == "__main__":
    main()