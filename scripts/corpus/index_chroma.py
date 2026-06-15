from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from rag_corpus.index import get_chroma_collection, index_chunks
from rag_corpus.manifest import write_manifest
from rag_corpus.paths import corpus_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index corpus chunks into ChromaDB.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--collection-name", default="research_papers")
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = corpus_paths(args.data_dir)
    
    chunks_file = paths.chunks_jsonl
    if not chunks_file.exists():
        print(f"Error: Chunks file not found at {chunks_file}. Please run chunk_corpus.py first.")
        sys.exit(1)
        
    print(f"Loading chunks from {chunks_file}...")
    chunks = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    total_chunks = len(chunks)
    print(f"Loaded {total_chunks} chunks.")
    
    chroma_db_dir = paths.processed_dir / "chroma_db"
    print(f"Initializing ChromaDB persistent store at {chroma_db_dir}...")
    collection = get_chroma_collection(chroma_db_dir, args.collection_name)
    
    print(f"Indexing chunks into collection '{args.collection_name}'...")
    index_chunks(collection, chunks, batch_size=args.batch_size)
    
    manifest_path = write_manifest(
        paths,
        step="index_chroma",
        parameters={
            "collection_name": args.collection_name,
            "chroma_db_dir": str(chroma_db_dir),
            "chunks_file": str(chunks_file),
            "batch_size": args.batch_size,
        },
        metrics={
            "chunks_indexed": total_chunks,
        }
    )
    print(f"ChromaDB indexing complete. Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
