from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from rag_corpus.paths import corpus_paths
from rag_corpus.retrieval import (
    BM25Searcher,
    format_context_prompt,
    generate_answer,
    get_overlapping_paragraphs,
    get_chroma_vectorstore,
    load_documents,
    reciprocal_rank_fusion,
    re_rank_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument("--query", required=True, help="User query string")
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--collection-name", default="research_papers")
    parser.add_argument("--provider", choices=["gemini", "openai", "mock"], default=None)
    parser.add_argument("--model", default=None, help="LLM model name to override default")
    parser.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid",
                        help="Retrieval mode: vector (semantic), bm25 (keyword), or hybrid (both via RRF)")
    parser.add_argument("--rrf-k", type=int, default=60, help="Constant for Reciprocal Rank Fusion")
    parser.add_argument("--re-rank", action="store_true", help="Enable second-stage re-ranking using Cohere")
    parser.add_argument("--re-rank-model", default="rerank-english-v3.0", help="Cohere reranker model name")
    parser.add_argument("--pool-size", type=int, default=25, help="Number of candidate chunks to fetch before re-ranking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = corpus_paths(args.data_dir)
    
    chroma_db_dir = paths.processed_dir / "chroma_db"
    if not chroma_db_dir.exists():
        print(f"Error: ChromaDB directory not found at {chroma_db_dir}. Please run index_chroma.py first.")
        sys.exit(1)
        
    print(f"Loading documents from {paths.documents_jsonl}...")
    docs = load_documents(paths.documents_jsonl)
    print(f"Loaded {len(docs)} documents into memory.")
    
    # Load chunks for BM25 if needed
    chunks = []
    if args.mode in ("bm25", "hybrid"):
        if not paths.chunks_jsonl.exists():
            print(f"Error: Chunks file not found at {paths.chunks_jsonl}. Cannot run BM25 search.")
            sys.exit(1)
        import json
        print(f"Loading chunks from {paths.chunks_jsonl} for BM25...")
        with paths.chunks_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        print(f"Loaded {len(chunks)} chunks for BM25.")
        
    bm25_searcher = None
    if args.mode in ("bm25", "hybrid"):
        print(f"Building in-memory BM25 index...")
        bm25_searcher = BM25Searcher(chunks)
        
    candidates = []
    fetch_k = args.pool_size if args.re_rank else args.k
    
    if args.mode == "vector":
        print(f"Connecting to ChromaDB at {chroma_db_dir} using LangChain...")
        vectorstore = get_chroma_vectorstore(chroma_db_dir, args.collection_name)
        print(f"Retrieving top {fetch_k} chunks using vector search for query: '{args.query}'...")
        results = vectorstore.similarity_search_with_score(args.query, k=fetch_k)
        for doc, dist in results:
            candidates.append({
                "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                "text": doc.page_content,
                "distance": float(dist),
                "metadata": {k: v for k, v in doc.metadata.items() if k not in ("chunk_id", "document_id", "chunk_index")},
                "document_id": doc.metadata["document_id"],
                "chunk_index": doc.metadata["chunk_index"]
            })
                
    elif args.mode == "bm25":
        print(f"Retrieving top {fetch_k} chunks using BM25 search for query: '{args.query}'...")
        bm25_results = bm25_searcher.search(args.query, top_n=fetch_k)
        for chunk in bm25_results:
            candidates.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "bm25_score": chunk["bm25_score"],
                "metadata": chunk["metadata"],
            })
            
    elif args.mode == "hybrid":
        # 1. Vector retrieval
        print(f"Connecting to ChromaDB at {chroma_db_dir} using LangChain...")
        vectorstore = get_chroma_vectorstore(chroma_db_dir, args.collection_name)
        print(f"Retrieving top {fetch_k} chunks using vector search...")
        v_results = vectorstore.similarity_search_with_score(args.query, k=fetch_k)
        vector_chunks = []
        for doc, dist in v_results:
            vector_chunks.append({
                "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                "text": doc.page_content,
                "distance": float(dist),
                "metadata": {k: v for k, v in doc.metadata.items() if k not in ("chunk_id", "document_id", "chunk_index")},
                "document_id": doc.metadata["document_id"],
                "chunk_index": doc.metadata["chunk_index"]
            })
                
        # 2. BM25 retrieval
        print(f"Retrieving top {fetch_k} chunks using BM25 search...")
        bm25_chunks = bm25_searcher.search(args.query, top_n=fetch_k)
        
        # 3. Handle Fusion or Union based on re-ranking
        if not args.re_rank:
            print(f"Fusing top {args.k} results using Reciprocal Rank Fusion (rrf_k={args.rrf_k})...")
            # Slice top k for standard RRF
            fused_chunks = reciprocal_rank_fusion(vector_chunks[:args.k], bm25_chunks[:args.k], rrf_k=args.rrf_k)
            candidates = fused_chunks
        else:
            # Union & de-duplicate the larger pool size
            merged = {}
            for chunk in vector_chunks:
                merged[chunk["chunk_id"]] = chunk
            for chunk in bm25_chunks:
                cid = chunk["chunk_id"]
                if cid not in merged:
                    merged[cid] = chunk
                else:
                    merged[cid]["bm25_score"] = chunk["bm25_score"]
            candidates = list(merged.values())
            
    # Perform second-stage Cohere re-ranking if requested
    retrieved_chunks = []
    if args.re_rank:
        if not candidates:
            print("No candidates retrieved to re-rank.")
        else:
            print(f"Re-ranking {len(candidates)} candidates using Cohere Re-rank API (model: {args.re_rank_model})...")
            try:
                retrieved_chunks = re_rank_chunks(
                    query=args.query,
                    chunks=candidates,
                    top_n=args.k,
                    model=args.re_rank_model
                )
            except Exception as e:
                print(f"Error during Cohere re-ranking: {e}")
                print("Falling back to initial retrieval rankings...")
                if args.mode == "hybrid":
                    print(f"Fusing top {args.k} results using Reciprocal Rank Fusion fallback...")
                    fused_chunks = reciprocal_rank_fusion(vector_chunks[:args.k], bm25_chunks[:args.k], rrf_k=args.rrf_k)
                    retrieved_chunks = fused_chunks[:args.k]
                else:
                    retrieved_chunks = candidates[:args.k]
    else:
        retrieved_chunks = candidates[:args.k]
        
    # Check if we got any results
    if not retrieved_chunks:
        print("No relevant chunks found.")
        sys.exit(0)
        
    retrieved_contexts = []
    
    print("\n--- RETRIEVED CHUNKS ---")
    for rank, chunk in enumerate(retrieved_chunks, 1):
        chunk_id = chunk["chunk_id"]
        meta = chunk.get("metadata", {})
        doc_id = meta.get("document_id", chunk_id.split("::")[0])
        text = chunk["text"]
        
        score_info = []
        if "distance" in chunk:
            score_info.append(f"Vector Dist: {chunk['distance']:.4f}")
        if "bm25_score" in chunk:
            score_info.append(f"BM25 Score: {chunk['bm25_score']:.4f}")
        if "rrf_score" in chunk:
            score_info.append(f"RRF Score: {chunk['rrf_score']:.4f}")
        if "cohere_rerank_score" in chunk:
            score_info.append(f"Cohere Rerank Score: {chunk['cohere_rerank_score']:.4f}")
            
        score_str = ", ".join(score_info)
        print(f"\n[{rank}] Chunk: {chunk_id} ({score_str})")
        
        # Map back to exact page or paragraph in the source document
        doc = docs.get(doc_id)
        if doc:
            overlapping_paras = get_overlapping_paragraphs(text, doc)
            if overlapping_paras:
                for p in overlapping_paras:
                    citation_id = f"{doc_id}, {p['label']}"
                    print(f"  |-> Matched Citation: [{citation_id}]")
                    # Add to context
                    retrieved_contexts.append({
                        "citation_id": citation_id,
                        "text": p["text"],
                        "metadata": meta,
                    })
            else:
                # Fallback to chunk metadata if no specific page/paragraph matched
                citation_id = f"{doc_id}, Chunk {meta.get('chunk_index', 0)}"
                print(f"  |-> (Fallback) Matched Citation: [{citation_id}]")
                retrieved_contexts.append({
                    "citation_id": citation_id,
                    "text": text,
                    "metadata": meta,
                })
        else:
            # Fallback if document not found in cache
            citation_id = f"{doc_id}, Chunk {meta.get('chunk_index', 0)}"
            print(f"  |-> (Doc Missing) Matched Citation: [{citation_id}]")
            retrieved_contexts.append({
                "citation_id": citation_id,
                "text": text,
                "metadata": meta,
            })
            
    # Remove duplicate contexts to prevent prompt inflation
    unique_contexts = []
    seen_ids = set()
    for ctx in retrieved_contexts:
        if ctx["citation_id"] not in seen_ids:
            seen_ids.add(ctx["citation_id"])
            unique_contexts.append(ctx)
            
    print(f"\nFormatted {len(unique_contexts)} unique paragraphs as LLM context.")
    
    # Format and generate
    prompt = format_context_prompt(args.query, unique_contexts)
    
    print("\n--- GENERATING ANSWER ---")
    try:
        answer, provider_info = generate_answer(prompt, unique_contexts, provider=args.provider, model=args.model)
        print(f"Reader Provider: {provider_info}")
        print("\n=== ANSWER ===")
        print(answer)
        print("==============")
    except Exception as e:
        print(f"Error generating answer: {e}")
        print("\nRaw Context Prompt constructed:")
        print(prompt[:500] + "...\n[prompt truncated]")


if __name__ == "__main__":
    main()
