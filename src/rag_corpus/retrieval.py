from __future__ import annotations

import json
import os
from pathlib import Path
import re

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from pydantic import BaseModel, Field


def tokenize(text: str) -> list[str]:
    """
    Lowercase and tokenize text into words.
    """
    return re.findall(r"\w+", text.lower())


class BGEEmbeddings(Embeddings):
    """
    BAAI/bge-large-en-v1.5 embedding wrapper.
    1024-dim, cosine-similarity optimised, top-ranked on MTEB retrieval.

    BGE uses an asymmetric retrieval pattern:
    - Queries get a special instruction prefix to boost retrieval accuracy.
    - Documents are encoded as-is (no prefix).
    - normalize_embeddings=True makes cosine similarity = dot product (faster, stable).
    """
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [self.QUERY_INSTRUCTION + text],
            normalize_embeddings=True
        )[0].tolist()


# Global cached instance for BGEEmbeddings to prevent loading the PyTorch model on every request
_bge_embeddings_instance = None

def get_bge_embeddings() -> BGEEmbeddings:
    global _bge_embeddings_instance
    if _bge_embeddings_instance is None:
        _bge_embeddings_instance = BGEEmbeddings()
    return _bge_embeddings_instance


def get_chroma_vectorstore(persist_directory: Path, collection_name: str) -> Chroma:
    """
    Load Chroma vector store using LangChain with BGE-large embeddings and cosine distance.
    """
    embeddings = get_bge_embeddings()
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )


class BM25Searcher:
    """
    BM25 Searcher using rank_bm25 directly for accurate relevance scores.
    LangChain's BM25Retriever.invoke() discards scores — we bypass it and
    call the underlying vectorizer to get real BM25 scores.
    """
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        docs = [
            Document(
                page_content=c["text"],
                metadata={
                    **c.get("metadata", {}),
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "chunk_index": c["chunk_index"]
                }
            )
            for c in chunks
        ]
        self.retriever = BM25Retriever.from_documents(docs, preprocess_func=tokenize)

    def search(self, query: str, top_n: int = 10) -> list[dict]:
        if not self.chunks:
            return []

        # Tokenize query with the same function used at index time
        tokenized_query = tokenize(query)

        # Access the underlying rank_bm25 object to get real scores
        all_scores = self.retriever.vectorizer.get_scores(tokenized_query)

        # Pair each chunk with its score, sort descending, take top_n
        scored = sorted(
            zip(all_scores, self.chunks),
            key=lambda x: x[0],
            reverse=True
        )[:top_n]

        results = []
        for score, chunk in scored:
            results.append({
                "chunk_id":    chunk["chunk_id"],
                "text":        chunk["text"],
                "metadata":    {k: v for k, v in chunk.get("metadata", {}).items()},
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "bm25_score":  float(score),
            })
        return results



def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    rrf_k: int = 60
) -> list[dict]:
    """
    Combine list of chunks retrieved via vector search and BM25 search
    using Reciprocal Rank Fusion (RRF).
    """
    scores = {}
    chunk_map = {}
    
    # Track rank in vector search results
    for rank, chunk in enumerate(vector_results, 1):
        chunk_id = chunk["chunk_id"]
        chunk_map[chunk_id] = chunk
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        
    # Track rank in BM25 search results
    for rank, chunk in enumerate(bm25_results, 1):
        chunk_id = chunk["chunk_id"]
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = chunk
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        
    # Sort IDs by RRF score descending
    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    ranked_chunks = []
    for chunk_id, score in sorted_ids:
        chunk = dict(chunk_map[chunk_id])
        chunk["rrf_score"] = score
        ranked_chunks.append(chunk)
        
    return ranked_chunks


def re_rank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 4,
    model: str = "rerank-english-v3.0",
    api_key: str | None = None
) -> list[dict]:
    """
    Re-rank a list of retrieved chunks using LangChain's CohereRerank.
    """
    if not chunks:
        return []
        
    cohere_key = api_key or os.environ.get("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError(
            "COHERE_API_KEY environment variable is not set. "
            "Please set it or provide it to enable Cohere re-ranking."
        )
        
    # Convert chunks to Document list
    docs = [
        Document(
            page_content=c["text"],
            metadata={
                **c.get("metadata", {}),
                "chunk_id": c["chunk_id"]
            }
        )
        for c in chunks
    ]
    
    compressor = CohereRerank(model=model, cohere_api_key=cohere_key, top_n=top_n)
    compressed_docs = compressor.compress_documents(documents=docs, query=query)
    
    re_ranked = []
    for doc in compressed_docs:
        chunk_id = doc.metadata.get("chunk_id")
        orig_chunk = next((c for c in chunks if c["chunk_id"] == chunk_id), None)
        if orig_chunk:
            chunk_copy = dict(orig_chunk)
            chunk_copy["cohere_rerank_score"] = float(doc.metadata.get("relevance_score", 0.0))
            re_ranked.append(chunk_copy)
            
    return re_ranked


def load_documents(documents_file: Path) -> dict[str, dict]:
    """
    Load all source documents from documents.jsonl into a dictionary keyed by document_id.
    """
    docs = {}
    if not documents_file.exists():
        return docs
    with documents_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                docs[doc["document_id"]] = doc
    return docs


def get_document_paragraphs(doc: dict) -> list[dict[str, str]]:
    """
    Split a document into logical sections (Title, Abstract, Page 1, Page 2, etc.)
    using structural metadata and newlines.
    """
    paragraphs = []
    if doc.get("title"):
        paragraphs.append({"label": "Title", "text": doc["title"].strip()})
    if doc.get("abstract"):
        paragraphs.append({"label": "Abstract", "text": doc["abstract"].strip()})
        
    text = doc.get("text", "")
    if text:
        pages = text.split("\n\n")
        for idx, page in enumerate(pages, 1):
            if page.strip():
                paragraphs.append({"label": f"Page {idx}", "text": page.strip()})
    return paragraphs


def get_overlapping_paragraphs(chunk_text: str, doc: dict) -> list[dict[str, str]]:
    """
    Find which paragraphs/pages in the source document overlap with the retrieved chunk text.
    """
    paragraphs = get_document_paragraphs(doc)
    norm_chunk = " ".join(chunk_text.split())
    overlapping = []
    
    for p in paragraphs:
        norm_p = " ".join(p["text"].split())
        if len(norm_p) < 50:
            if norm_p in norm_chunk:
                overlapping.append(p)
        else:
            p_prefix = norm_p[:50]
            p_suffix = norm_p[-50:]
            if p_prefix in norm_chunk or p_suffix in norm_chunk or norm_chunk in norm_p:
                overlapping.append(p)
                
    return overlapping


def load_prompt_config() -> dict:
    """
    Load the versioned prompt config file relative to the project root.
    """
    config_path = Path(__file__).resolve().parents[2] / "config" / "prompts.json"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    return {
        "rag_system_prompt": (
            "You are an expert assistant. Your task is to answer the query using ONLY the provided references.\n\n"
            "You MUST return a JSON object with the following structure:\n"
            "{\n  \"supported\": true,\n  \"answer\": \"your answer with citations, e.g. [1]\"\n}\n\n"
            "Rules:\n"
            "1. If the provided references do not contain enough information to answer the query fully and directly, "
            "you MUST set \"supported\" to false, and set \"answer\" to exactly: \"I do not have enough information to answer this query.\"\n"
            "2. Do not use any outside knowledge.\n"
            "3. Every statement must be explicitly cited in square brackets."
        ),
        "decline_message": "I do not have enough information to answer this query based on the provided documents."
    }


# Max characters per context passage sent to the LLM.
# Keeps the prompt tight so the model focuses on the most relevant text.
MAX_CTX_CHARS = 8000


def format_context_prompt(query: str, retrieved_contexts: list[dict]) -> str:
    """
    Construct the context prompt for the generative reader.
    Each context is capped at MAX_CTX_CHARS to avoid prompt bloat.
    """
    context_str = ""
    for idx, ctx in enumerate(retrieved_contexts, 1):
        text = ctx["text"].strip()
        # Trim to cap, breaking at a sentence boundary where possible
        if len(text) > MAX_CTX_CHARS:
            trimmed = text[:MAX_CTX_CHARS]
            last_period = trimmed.rfind(".")
            if last_period > MAX_CTX_CHARS // 2:
                trimmed = trimmed[: last_period + 1]
            text = trimmed + " [...]"

        context_str += f"[{idx}] {ctx['citation_id']}\n{text}\n" + "-" * 40 + "\n"

    config = load_prompt_config()
    system_prompt = config.get("rag_system_prompt")

    prompt = (
        f"{system_prompt}\n\n"
        f"Query: {query}\n\n"
        f"References:\n{context_str}\n"
        "Answer:"
    )
    return prompt


# A small set of standard sentence-starting words to ignore if capitalized at the beginning of a sentence.
SENTENCE_START_WORDS = {
    "the", "this", "it", "in", "a", "an", "they", "we", "he", "she", 
    "here", "there", "to", "for", "as", "our", "their", "moreover", 
    "however", "additionally", "furthermore", "consequently", "therefore",
    "thus", "hence", "instead", "similarly"
}


def extract_grounding_candidates(text: str) -> set[str]:
    """
    Extract key factual candidates (Proper Nouns, Acronyms, and Multi-digit Numbers) 
    from the text that must be grounded in the context.
    """
    # Strip citation patterns like [1], [12], [1-3], [1, 2] to prevent citation numbers from being treated as candidates
    clean_text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", text)
    clean_text = re.sub(r"\[\d+\s*-\s*\d+\]", " ", clean_text)

    candidates = set()
    # Split text into sentences using simple punctuation split
    sentences = re.split(r"[.!?]\s+", clean_text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Find all words in the sentence (including hyphenated ones)
        words = re.findall(r"\b[A-Za-z0-9_-]+\b", sentence)
        if not words:
            continue
            
        for idx, word in enumerate(words):
            # Check if it is a number of length >= 2 (ignoring single digits like citation indexes)
            if re.match(r"^\d+$", word):
                if len(word) >= 2:
                    candidates.add(word)
                continue
                
            # Check if it is capitalized (Proper Noun or Acronym)
            if word and word[0].isupper():
                # If it's the first word of the sentence, ignore unless it is all-caps or mixed-case (e.g. system/acronym name)
                if idx == 0:
                    is_special_case = word.isupper() or any(c.isupper() for c in word[1:])
                    if not is_special_case:
                        continue
                candidates.add(word.lower())
                
    return candidates



def is_candidate_grounded(candidate: str, context_words: set[str]) -> bool:
    """
    Check if a candidate word/number is grounded in the retrieved context words.
    Supports singular/plural mapping and prefix checks.
    """
    # 1. Direct case-insensitive match
    if candidate in context_words:
        return True
        
    # 2. Singular/plural match (e.g. "llms" -> "llm")
    if candidate.endswith("s") and len(candidate) > 3:
        if candidate[:-1] in context_words:
            return True
            
    # 3. Hyphenated word breakdown (e.g. "toulmin-style" -> parts check)
    if "-" in candidate:
        parts = candidate.split("-")
        if all(p in context_words or len(p) < 3 for p in parts if p):
            return True
            
    # 4. Fuzzy prefix match (e.g. "falsifybench" starts with/starts the context word)
    for cw in context_words:
        if len(candidate) >= 4 and len(cw) >= 4:
            if candidate.startswith(cw) or cw.startswith(candidate):
                return True
                
    # 5. Substring match (e.g. "giou" in "lgiou")
    for cw in context_words:
        if len(candidate) >= 3 and candidate in cw:
            return True
                
    return False


class RAGResponse(BaseModel):
    supported: bool = Field(
        description="True if the references contain enough information to directly answer the query, else False."
    )
    answer: str = Field(
        description="The detailed answer to the query. EVERY statement in the answer MUST be cited using the reference number in square brackets (e.g. [1], [2]). If unsupported, return the decline message."
    )


def extract_answer_sentences(
    query: str,
    retrieved_contexts: list[dict],
    top_k: int = 5,
    min_len: int = 40,
) -> tuple[str, str]:
    """
    Extractive QA — no LLM required.

    Scores every sentence from the retrieved chunks against the query using
    sentence-level BM25 (rank_bm25), picks the top-k most relevant sentences,
    deduplicates, and returns them stitched together with [N] citation markers.
    """
    from rank_bm25 import BM25Okapi

    # ── Build sentence pool from all retrieved contexts ──────────────────────
    sentence_pool: list[dict] = []
    for ctx_idx, ctx in enumerate(retrieved_contexts, 1):
        raw = ctx.get("text", "")
        # Split on sentence-ending punctuation followed by whitespace / end
        sentences = re.split(r"(?<=[.!?])\s+", raw)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= min_len:
                sentence_pool.append({
                    "text":         sent,
                    "citation_idx": ctx_idx,
                })

    if not sentence_pool:
        return (
            "No relevant content was found in the knowledge base for this query.",
            "Extractive Reader",
        )

    # ── Score sentences with BM25 ───────────────────────────────────────
    tokenized_corpus = [tokenize(s["text"]) for s in sentence_pool]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize(query))

    # ── Select top-k, deduplicate by prefix ──────────────────────────────
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    seen_prefixes: set[str] = set()
    selected: list[dict] = []
    for idx, score in ranked:
        if len(selected) >= top_k:
            break
        if score <= 0:
            break
        sent = sentence_pool[idx]
        prefix = sent["text"][:60].lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        selected.append({**sent, "score": score})

    if not selected:
        return (
            "No sufficiently relevant sentences were found for this query.",
            "Extractive Reader",
        )

    # ── Sort by source order for coherent reading, then format ─────────────
    selected.sort(key=lambda s: s["citation_idx"])
    parts = [f"{s['text']} [{s['citation_idx']}]" for s in selected]
    return " ".join(parts), "Extractive Reader"



def generate_answer(
    prompt: str,
    retrieved_contexts: list[dict],
    provider: str | None = None,
    model: str | None = None
) -> tuple[str, str]:
    """
    Generate an answer from retrieved contexts.
    provider options: 'extractive' (default, no API), 'gemini', 'openai'.
    """
    config = load_prompt_config()
    decline_message = config.get("decline_message")

    if not retrieved_contexts:
        return decline_message, "Enforced Decline (Empty Context)"

    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Default: extractive when no provider specified and no LLM keys available
    if provider is None:
        if gemini_key:
            provider = "gemini"
        elif openai_key:
            provider = "openai"
        else:
            provider = "extractive"

    # ── Extractive (local, no API) ────────────────────────────────────────
    if provider == "extractive":
        query = prompt
        if "Query:" in prompt:
            query = prompt.split("Query:", 1)[1].split("\n")[0].strip()
        return extract_answer_sentences(query, retrieved_contexts)

    if provider == "gemini":
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = model or "gemini-3.1-flash-lite"
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_key,
            temperature=0.0
        )
        structured_llm = llm.with_structured_output(RAGResponse)

        system_prompt = config.get("rag_system_prompt")
        human_content = prompt
        if "Query:" in prompt:
            human_content = "Query:" + prompt.split("Query:", 1)[1]

        messages = [
            ("system", system_prompt),
            ("human", human_content)
        ]

        try:
            res = structured_llm.invoke(messages)
            supported = res.supported
            answer = res.answer
        except Exception as e:
            print(f"Gemini structured output failed: {e}")
            return decline_message, "Gemini"

        provider_info = "Gemini"

    elif provider == "openai":
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        from langchain_openai import ChatOpenAI
        
        model_name = model or "gpt-4o-mini"
        llm = ChatOpenAI(
            model=model_name,
            api_key=openai_key,
            temperature=0.0
        )
        structured_llm = llm.with_structured_output(RAGResponse)
        
        system_prompt = config.get("rag_system_prompt")
        human_content = prompt
        if "Query:" in prompt:
            parts = prompt.split("Query:", 1)
            human_content = "Query:" + parts[1]
            
        messages = [
            ("system", system_prompt),
            ("human", human_content)
        ]
        
        try:
            res = structured_llm.invoke(messages)
            supported = res.supported
            answer = res.answer
        except Exception as e:
            print(f"OpenAI Structured Output failed: {e}")
            return decline_message, f"OpenAI ({model_name}) (Structured Fallback Error)"
            
        provider_info = f"OpenAI ({model_name})"
        
    elif provider == "mock":
        sources_found = []
        for ctx in retrieved_contexts:
            doc_title = ctx.get("metadata", {}).get("title", "Unknown Title")
            citation_id = ctx["citation_id"]
            snippet = ctx["text"][:120] + "..." if len(ctx["text"]) > 120 else ctx["text"]
            sources_found.append(f"- **{citation_id}** ({doc_title}): \"{snippet}\"")
            
        mock_response = (
            "[MOCK GENERATOR MODE - No API keys configured]\n\n"
            "If an API key were configured, the generative reader would answer the query using these matched sources:\n"
            + "\n".join(sources_found) + "\n\n"
            "To activate full LLM answering with precise in-line citations, run:\n"
            "  $env:GEMINI_API_KEY=\"your-api-key\"\n"
            "or:\n"
            "  $env:OPENAI_API_KEY=\"your-api-key\""
        )
        return mock_response, "Mock Reader (Local Snippets)"
    else:
        raise ValueError(f"Unknown provider: {provider}")
        
    # ── Post-generation checks ────────────────────────────────────────────
    if provider in ("gemini", "openai"):
        if not supported:
            return decline_message, provider_info

        is_decline = (
            decline_message.lower() in answer.lower()
            or "not have enough information" in answer.lower()
            or "do not have enough information" in answer.lower()
        )
        if is_decline:
            return decline_message, provider_info

        # Grounding check to prevent hallucinations of entities/numbers
        context_words = set(tokenize(" ".join(ctx["text"] for ctx in retrieved_contexts)))
        candidates = extract_grounding_candidates(answer)
        ungrounded = [c for c in candidates if not is_candidate_grounded(c, context_words)]
        if ungrounded:
            print(f"[WARNING] LLM answer contains ungrounded/hallucinated terms: {ungrounded}. Declining response to prevent hallucination.")
            return decline_message, provider_info

    return answer, provider_info

