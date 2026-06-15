from __future__ import annotations

import json
from pathlib import Path
import chromadb
from chromadb.api.models.Collection import Collection


def clean_metadata(metadata: dict) -> dict:
    """
    Clean metadata dictionary to ensure all values are simple types (str, int, float, bool)
    supported by ChromaDB. Convert lists to comma-separated strings and dicts/sets to JSON strings.
    """
    cleaned = {}
    for k, v in metadata.items():
        if v is None:
            cleaned[k] = ""
        elif isinstance(v, list):
            cleaned[k] = ", ".join(str(item) for item in v)
        elif isinstance(v, (dict, set)):
            cleaned[k] = json.dumps(v)
        elif isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned


def get_chroma_collection(persist_directory: Path, collection_name: str) -> Collection:
    """
    Initialize the Chroma persistent client and get or create the collection.
    Uses BAAI/bge-large-en-v1.5 for embedding (handled externally).
    Sets cosine distance metric to match BGE's training objective.
    """
    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def index_chunks(collection: Collection, chunks: list[dict], batch_size: int = 100) -> None:
    """
    Batch index chunks into the Chroma collection.
    """
    from rag_corpus.retrieval import BGEEmbeddings
    embeddings_model = BGEEmbeddings()
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        documents = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [clean_metadata({**c["metadata"], "document_id": c["document_id"], "chunk_index": c["chunk_index"]}) for c in batch]
        
        # Pre-calculate embeddings using the BGE model to match dimensions (1024)
        embeddings = embeddings_model.embed_documents(documents)
        
        collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Indexed {min(i + batch_size, total)}/{total} chunks...", flush=True)

