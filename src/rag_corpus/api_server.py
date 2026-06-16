import os
import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import functools

# Setup sys.path to resolve src
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))


from rag_corpus.paths import corpus_paths
from rag_corpus.retrieval import (
    get_chroma_vectorstore,
    reciprocal_rank_fusion,
    re_rank_chunks,
    get_overlapping_paragraphs,
    format_context_prompt,
    generate_answer,
    load_documents,
)

# Caching for document lists and BM25 searchers to optimize performance
@functools.lru_cache(maxsize=10)
def get_cached_documents(data_dir_str: str) -> dict:
    paths = corpus_paths(data_dir_str)
    return load_documents(paths.documents_jsonl)

@functools.lru_cache(maxsize=10)
def get_cached_bm25_searcher(data_dir_str: str):
    from rag_corpus.retrieval import BM25Searcher
    paths = corpus_paths(data_dir_str)
    chunks = []
    if paths.chunks_jsonl.exists():
        with paths.chunks_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
    return BM25Searcher(chunks)

app = FastAPI(
    title="Enterprise RAG API Server", 
    description="Backend server for LangChain RAG pipeline",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    data_dir: str = "data/llm_ai_2023_2026"
    collection_name: str = "research_papers"
    mode: str = "hybrid"
    re_rank: bool = True
    k: int = 5
    pool_size: int = 10
    rrf_k: int = 60
    answer_mode: str = "gemini"   # 'extractive' | 'gemini' | 'openai'
    cohere_key: str | None = None
    gemini_key: str | None = None

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/query")
def query_rag(req: QueryRequest):
    try:
        paths = corpus_paths(req.data_dir)
        chroma_db_dir = paths.processed_dir / "chroma_db"
        if not chroma_db_dir.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"ChromaDB directory not found at {chroma_db_dir}. Please run indexing first."
            )
            
        docs = get_cached_documents(req.data_dir)
        
        fetch_k = req.pool_size if (req.re_rank or req.mode == "hybrid") else req.k
        candidates = []
        
        # 1. Vector Search
        vector_chunks = []
        if req.mode in ("vector", "hybrid"):
            vectorstore = get_chroma_vectorstore(chroma_db_dir, req.collection_name)
            v_results = vectorstore.similarity_search_with_score(req.query, k=fetch_k)
            for doc, dist in v_results:
                vector_chunks.append({
                    "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                    "text": doc.page_content,
                    "distance": float(dist),
                    "metadata": {k: v for k, v in doc.metadata.items() if k not in ("chunk_id", "document_id", "chunk_index")},
                    "document_id": doc.metadata["document_id"],
                    "chunk_index": doc.metadata["chunk_index"]
                })
                
        # 2. BM25 Search
        bm25_chunks = []
        if req.mode in ("bm25", "hybrid"):
            bm25_searcher = get_cached_bm25_searcher(req.data_dir)
            bm25_chunks = bm25_searcher.search(req.query, top_n=fetch_k)
            
        # Combine/Select Candidates
        if req.mode == "vector":
            candidates = vector_chunks
        elif req.mode == "bm25":
            candidates = bm25_chunks
        elif req.mode == "hybrid":
            if not req.re_rank:
                # Direct RRF
                candidates = reciprocal_rank_fusion(vector_chunks, bm25_chunks, rrf_k=req.rrf_k)
            else:
                # Merge & deduplicate to create a pool for reranking
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
                
        # 3. Second-stage Cohere re-ranking
        retrieved_chunks = []
        cohere_key = req.cohere_key or os.environ.get("COHERE_API_KEY")
        if req.re_rank:
            if not candidates:
                retrieved_chunks = []
            elif not cohere_key:
                print("Warning: Cohere re-ranking requested but no key provided. Falling back to RRF...")
                if req.mode == "hybrid":
                    retrieved_chunks = reciprocal_rank_fusion(vector_chunks[:req.k], bm25_chunks[:req.k], rrf_k=req.rrf_k)
                else:
                    retrieved_chunks = candidates[:req.k]
            else:
                try:
                    retrieved_chunks = re_rank_chunks(
                        query=req.query,
                        chunks=candidates,
                        top_n=req.k,
                        api_key=cohere_key
                    )
                except Exception as e:
                    print(f"Error during Cohere re-ranking: {e}. Falling back to initial retrieval rankings...")
                    if req.mode == "hybrid":
                        retrieved_chunks = reciprocal_rank_fusion(vector_chunks[:req.k], bm25_chunks[:req.k], rrf_k=req.rrf_k)
                    else:
                        retrieved_chunks = candidates[:req.k]
        else:
            retrieved_chunks = candidates[:req.k]
            
        # Map chunks back to exact paragraphs/pages in document
        retrieved_contexts = []
        for chunk in retrieved_chunks:
            chunk_id = chunk["chunk_id"]
            meta = chunk.get("metadata", {})
            doc_id = meta.get("document_id", chunk_id.split("::")[0])
            text = chunk["text"]
            
            doc = docs.get(doc_id)
            if doc:
                overlapping_paras = get_overlapping_paragraphs(text, doc)
                if overlapping_paras:
                    for p in overlapping_paras:
                        retrieved_contexts.append({
                            "citation_id": f"{doc_id}, {p['label']}",
                            "text": p["text"],
                            "metadata": meta,
                            "chunk_id": chunk_id,
                            "relevance_info": {k: v for k, v in chunk.items() if k not in ("text", "metadata")}
                        })
                else:
                    retrieved_contexts.append({
                        "citation_id": f"{doc_id}, Chunk {meta.get('chunk_index', 0)}",
                        "text": text,
                        "metadata": meta,
                        "chunk_id": chunk_id,
                        "relevance_info": {k: v for k, v in chunk.items() if k not in ("text", "metadata")}
                    })
            else:
                retrieved_contexts.append({
                    "citation_id": f"{doc_id}, Chunk {meta.get('chunk_index', 0)}",
                    "text": text,
                    "metadata": meta,
                    "chunk_id": chunk_id,
                    "relevance_info": {k: v for k, v in chunk.items() if k not in ("text", "metadata")}
                })
                
        # Remove duplicate contexts to prevent prompt inflation
        unique_contexts = []
        seen_ids = set()
        for ctx in retrieved_contexts:
            if ctx["citation_id"] not in seen_ids:
                seen_ids.add(ctx["citation_id"])
                unique_contexts.append(ctx)
                
        # Generate LLM response
        prompt = format_context_prompt(req.query, unique_contexts)
        
        # Inject API keys if provided from settings overrides
        old_gemini = os.environ.get("GEMINI_API_KEY")
        if req.gemini_key:
            os.environ["GEMINI_API_KEY"] = req.gemini_key
            
        old_cohere = os.environ.get("COHERE_API_KEY")
        if req.cohere_key:
            os.environ["COHERE_API_KEY"] = req.cohere_key
            
        # Determine active key to select provider
        active_gemini_key = req.gemini_key or os.environ.get("GEMINI_API_KEY")

        # Map answer_mode to provider, falling back to extractive if key missing
        provider = req.answer_mode  # 'extractive', 'gemini', 'openai'
        if provider == "gemini" and not active_gemini_key:
            provider = "extractive"

        answer, provider_info = generate_answer(
            prompt=prompt,
            retrieved_contexts=unique_contexts,
            provider=provider
        )
        
        # Revert environment keys
        if req.gemini_key and old_gemini is not None:
            os.environ["GEMINI_API_KEY"] = old_gemini
        elif req.gemini_key:
            os.environ.pop("GEMINI_API_KEY", None)
            
        if req.cohere_key and old_cohere is not None:
            os.environ["COHERE_API_KEY"] = old_cohere
        elif req.cohere_key:
            os.environ.pop("COHERE_API_KEY", None)
            
        return {
            "answer": answer,
            "provider_info": provider_info,
            "retrieved_chunks": retrieved_chunks,
            "contexts": unique_contexts
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
