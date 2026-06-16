#!/usr/bin/env python3
"""
evaluate_faithfulness.py
========================
Runs Ragas Faithfulness + AnswerRelevancy evaluation over your RAG pipeline.

Usage:
    $env:GEMINI_API_KEY = "your-key"
    .\venv\Scripts\python.exe scripts\evaluate_faithfulness.py

    # Limit to first N questions for a quick test:
    .\venv\Scripts\python.exe scripts\evaluate_faithfulness.py --limit 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Workaround for Ragas 0.4.3 ModuleNotFoundError: No module named
# 'langchain_community.chat_models.vertexai'
mock_module = MagicMock()
mock_module.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = mock_module

import requests
from datasets import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA_JSON = PROJECT_ROOT / "data" / "evaluation" / "qa_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_API_URL = "http://localhost:8000/api/query"

RAG_SETTINGS = {
    "data_dir": "data/llm_ai_2023_2026",
    "mode": "hybrid",
    "re_rank": False,
    "k": 5,
    "pool_size": 15,
    "answer_mode": "gemini",
}


def call_rag(question: str, gemini_key: str, api_url: str) -> dict | None:
    """Call the RAG API and return the full response dict."""
    payload = {
        **RAG_SETTINGS,
        "query": question,
        "gemini_key": gemini_key,
    }
    try:
        resp = requests.post(api_url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"  [ERROR] API call failed: {exc}")
        return None


def build_ragas_dataset(
    qa_entries: list[dict],
    gemini_key: str,
    api_url: str,
    limit: int = 0,
) -> tuple[Dataset, dict]:
    """
    Call the RAG backend for each question and collect
    (user_input, response, retrieved_contexts) triples.
    """
    if limit > 0:
        qa_entries = qa_entries[:limit]

    rows = {"user_input": [], "response": [], "retrieved_contexts": []}
    skipped_declined = 0
    skipped_no_context = 0
    skipped_api_error = 0

    for i, entry in enumerate(qa_entries, 1):
        question = entry["question"]
        print(f"[{i:3}/{len(qa_entries)}] {question[:90]}...")

        result = call_rag(question, gemini_key, api_url)
        if result is None:
            skipped_api_error += 1
            continue

        answer = result.get("answer", "")
        contexts = result.get("contexts", [])

        if "not have enough information" in answer.lower():
            print("  -> Skipped (RAG declined)")
            skipped_declined += 1
            continue

        context_texts = [ctx.get("text", "") for ctx in contexts if ctx.get("text")]
        if not context_texts:
            print("  -> Skipped (no retrieved contexts)")
            skipped_no_context += 1
            continue

        rows["user_input"].append(question)
        rows["response"].append(answer)
        rows["retrieved_contexts"].append(context_texts)

        print(f"  [OK] answer={len(answer)} chars, contexts={len(context_texts)}")
        time.sleep(0.5)

    stats = {
        "total_questions": len(qa_entries),
        "n_evaluated": len(rows["user_input"]),
        "skipped_declined": skipped_declined,
        "skipped_no_context": skipped_no_context,
        "skipped_api_error": skipped_api_error,
        "skipped_total": skipped_declined + skipped_no_context + skipped_api_error,
    }

    print(
        "\nBuilt dataset: "
        f"{stats['n_evaluated']} rows "
        f"({stats['skipped_total']} skipped: "
        f"{stats['skipped_declined']} declined, "
        f"{stats['skipped_no_context']} no-context, "
        f"{stats['skipped_api_error']} API errors)"
    )
    return Dataset.from_dict(rows), stats


def run_evaluation(dataset: Dataset, gemini_key: str) -> dict:
    """Run Ragas metrics on the collected dataset."""
    import asyncio
    import time

    import google.generativeai as genai
    from ragas import evaluate
    from ragas.embeddings import GoogleEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import AnswerRelevancy, Faithfulness

    def retry_on_429(fn):
        if asyncio.iscoroutinefunction(fn):

            async def async_wrapper(*args, **kwargs):
                for attempt in range(1, 11):
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as exc:
                        err_msg = str(exc)
                        is_rate_limit = any(
                            token in err_msg
                            for token in ["429", "RESOURCE_EXHAUSTED", "rate_limit", "Quota exceeded"]
                        )
                        is_transient = any(
                            token in err_msg
                            for token in ["500", "503", "Internal Server Error", "Service Unavailable", "timeout"]
                        )
                        if is_rate_limit or is_transient:
                            wait_time = 20 * attempt
                            print(
                                "  [Gemini API] Rate limit / transient error: "
                                f"{exc}. Retrying in {wait_time}s... (Attempt {attempt}/10)"
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            raise
                return await fn(*args, **kwargs)

            return async_wrapper

        def sync_wrapper(*args, **kwargs):
            for attempt in range(1, 11):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    err_msg = str(exc)
                    is_rate_limit = any(
                        token in err_msg
                        for token in ["429", "RESOURCE_EXHAUSTED", "rate_limit", "Quota exceeded"]
                    )
                    is_transient = any(
                        token in err_msg
                        for token in ["500", "503", "Internal Server Error", "Service Unavailable", "timeout"]
                    )
                    if is_rate_limit or is_transient:
                        wait_time = 20 * attempt
                        print(
                            "  [Gemini API] Rate limit / transient error: "
                            f"{exc}. Retrying in {wait_time}s... (Attempt {attempt}/10)"
                        )
                        time.sleep(wait_time)
                    else:
                        raise
            return fn(*args, **kwargs)

        return sync_wrapper

    genai.configure(api_key=gemini_key)
    client = genai.GenerativeModel("gemini-3.1-flash-lite")

    judge_llm = llm_factory(
        "gemini-3.1-flash-lite",
        provider="google",
        client=client,
        adapter="instructor",
    )

    original_create = judge_llm.client.chat.completions.create

    def wrapped_create(*args, **kwargs):
        kwargs.pop("model", None)
        return original_create(*args, **kwargs)

    judge_llm.client.chat.completions.create = wrapped_create
    judge_llm.generate = retry_on_429(judge_llm.generate)
    judge_llm.agenerate = retry_on_429(judge_llm.agenerate)

    judge_embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")
    judge_embeddings.embed_text = retry_on_429(judge_embeddings.embed_text)
    judge_embeddings.aembed_text = retry_on_429(judge_embeddings.aembed_text)
    judge_embeddings.embed_texts = retry_on_429(judge_embeddings.embed_texts)
    judge_embeddings.aembed_texts = retry_on_429(judge_embeddings.aembed_texts)

    GoogleEmbeddings.embed_query = GoogleEmbeddings.embed_text
    GoogleEmbeddings.embed_documents = GoogleEmbeddings.embed_texts
    GoogleEmbeddings.aembed_query = GoogleEmbeddings.aembed_text
    GoogleEmbeddings.aembed_documents = GoogleEmbeddings.aembed_texts

    judge_embeddings.embed_query = judge_embeddings.embed_text
    judge_embeddings.embed_documents = judge_embeddings.embed_texts
    judge_embeddings.aembed_query = judge_embeddings.aembed_text
    judge_embeddings.aembed_documents = judge_embeddings.aembed_texts

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
    ]

    from ragas.run_config import RunConfig

    run_config = RunConfig(max_workers=1, max_retries=10, max_wait=60)

    print("\nRunning Ragas evaluation (this calls Gemini per question)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )
    return results

def verify_dataset_alignment(qa_entries: list[dict], data_dir: str) -> None:
    """
    Verify that the source documents for the QA dataset are actually present in the indexed corpus.
    """
    import re
    
    documents_file = Path(data_dir) / "processed" / "documents.jsonl"
    if not documents_file.exists():
        print(f"[WARNING] Corpus documents file not found at {documents_file}. Cannot verify alignment.")
        return
        
    print(f"Verifying alignment between QA dataset and corpus at {documents_file}...")
    corpus_docs = []
    with documents_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus_docs.append(json.loads(line))
                
    # Extract identifiers from corpus documents
    # A corpus document might be a PDF (with pdf_path containing arxiv ID) or have a title containing it.
    corpus_identifiers = set()
    for doc in corpus_docs:
        # Check pdf_path
        pdf_path = doc.get("pdf_path") or ""
        # Match arxiv patterns in path, e.g., arxiv_2606.04691v1.pdf
        arxiv_match = re.search(r"arxiv_(\d+\.\d+)", pdf_path)
        if arxiv_match:
            corpus_identifiers.add(arxiv_match.group(1))
        
        # Check document_id or path
        path = doc.get("path") or ""
        arxiv_match = re.search(r"arxiv_(\d+\.\d+)", path)
        if arxiv_match:
            corpus_identifiers.add(arxiv_match.group(1))
            
        doc_id = doc.get("document_id") or ""
        arxiv_match = re.search(r"(\d+\.\d+)", doc_id)
        if arxiv_match:
            corpus_identifiers.add(arxiv_match.group(1))
            
    # Check alignment for each QA entry
    matched_count = 0
    mismatched_arxiv_ids = set()
    for entry in qa_entries:
        arxiv_id = entry.get("arxiv_id")
        if not arxiv_id:
            # If there's no arxiv_id, we can fall back to matching title/url if available
            continue
            
        # Standardize arxiv_id (e.g. remove versions if any)
        clean_arxiv_id = arxiv_id.strip()
        
        # Check if the clean_arxiv_id exists in any of our corpus identifiers
        is_matched = False
        if clean_arxiv_id in corpus_identifiers:
            is_matched = True
        else:
            # Fallback direct substring match
            for doc in corpus_docs:
                title = doc.get("title") or ""
                path = doc.get("path") or ""
                pdf_path = doc.get("pdf_path") or ""
                url = doc.get("url") or ""
                if (clean_arxiv_id in title or 
                    clean_arxiv_id in path or 
                    clean_arxiv_id in pdf_path or 
                    clean_arxiv_id in url):
                    is_matched = True
                    break
                    
        if is_matched:
            matched_count += 1
        else:
            mismatched_arxiv_ids.add(clean_arxiv_id)
            
    total_with_id = sum(1 for entry in qa_entries if entry.get("arxiv_id"))
    if total_with_id == 0:
        print("No arXiv IDs found in the QA dataset. Skipping alignment check.")
        return
        
    print(f"Alignment check: {matched_count}/{total_with_id} QA entries matched with corpus documents.")
    if matched_count == 0:
        print("\n" + "!" * 80)
        print("CRITICAL MISMATCH DETECTED:")
        print(f"The QA dataset is referencing arXiv papers, but the indexed corpus at {data_dir}")
        print("contains entirely different documents (e.g. web pages).")
        print("All queries will fail retrieval or be declined by the model's grounding rules.")
        print(f"Mismatched arXiv IDs in QA dataset: {sorted(list(mismatched_arxiv_ids))}")
        print("!" * 80 + "\n")
        raise SystemExit("Fatal: QA dataset is mismatched with the indexed corpus. Please check your data seeding / indexing steps.")
    elif matched_count < total_with_id:
        print(f"[WARNING] Some QA entries ({total_with_id - matched_count}) do not have corresponding documents in the index.")
        print(f"Missing arXiv IDs: {sorted(list(mismatched_arxiv_ids))}")


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N questions (0 = all)")
    parser.add_argument(
        "--dataset-path",
        default=os.environ.get("RAG_EVAL_DATASET_PATH", str(DEFAULT_QA_JSON)),
        help="Path to the Q&A dataset JSON file.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("RAG_EVAL_API_URL", DEFAULT_API_URL),
        help="RAG API query endpoint.",
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=float(os.environ.get("RAG_EVAL_MIN_FAITHFULNESS", "0.80")),
        help="Fail if mean faithfulness drops below this threshold.",
    )
    parser.add_argument(
        "--min-answer-relevancy",
        type=float,
        default=float(os.environ.get("RAG_EVAL_MIN_ANSWER_RELEVANCY", "0.75")),
        help="Fail if mean answer relevancy drops below this threshold.",
    )
    parser.add_argument(
        "--min-evaluated",
        type=int,
        default=int(os.environ.get("RAG_EVAL_MIN_EVALUATED", "1")),
        help="Fail if fewer than this many questions are successfully evaluated.",
    )
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        raise SystemExit(1)

    qa_json = Path(args.dataset_path)
    with qa_json.open("r", encoding="utf-8") as handle:
        qa_entries = json.load(handle)
    print(f"Loaded {len(qa_entries)} Q&A pairs from {qa_json}")

    verify_dataset_alignment(qa_entries, RAG_SETTINGS["data_dir"])

    dataset, dataset_stats = build_ragas_dataset(
        qa_entries,
        gemini_key,
        api_url=args.api_url,
        limit=args.limit,
    )
    if len(dataset) == 0:
        print("No valid rows collected - check that the API is running.")
        raise SystemExit(1)

    results = run_evaluation(dataset, gemini_key)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = results.to_pandas()

    out_csv = RESULTS_DIR / "ragas_results.csv"
    out_json = RESULTS_DIR / "ragas_summary.json"
    df.to_csv(out_csv, index=False)

    summary = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "n_evaluated": len(df),
        "total_questions": dataset_stats["total_questions"],
        "skipped_total": dataset_stats["skipped_total"],
        "skipped_declined": dataset_stats["skipped_declined"],
        "skipped_no_context": dataset_stats["skipped_no_context"],
        "skipped_api_error": dataset_stats["skipped_api_error"],
        "thresholds": {
            "min_faithfulness": args.min_faithfulness,
            "min_answer_relevancy": args.min_answer_relevancy,
            "min_evaluated": args.min_evaluated,
        },
        "low_faithfulness": df[df["faithfulness"] < 0.5]["user_input"].tolist(),
    }
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    gate_failures = []
    if summary["faithfulness"] < args.min_faithfulness:
        gate_failures.append(
            f"faithfulness {summary['faithfulness']:.3f} < {args.min_faithfulness:.3f}"
        )
    if summary["answer_relevancy"] < args.min_answer_relevancy:
        gate_failures.append(
            f"answer_relevancy {summary['answer_relevancy']:.3f} < {args.min_answer_relevancy:.3f}"
        )
    if summary["n_evaluated"] < args.min_evaluated:
        gate_failures.append(f"n_evaluated {summary['n_evaluated']} < {args.min_evaluated}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(
        "  Faithfulness     : {:.3f}  (target >= {:.2f})".format(
            summary["faithfulness"], args.min_faithfulness
        )
    )
    print(
        "  Answer Relevancy : {:.3f}  (target >= {:.2f})".format(
            summary["answer_relevancy"], args.min_answer_relevancy
        )
    )
    print(f"  Questions scored : {summary['n_evaluated']}")
    print(f"  Questions total  : {summary['total_questions']}")
    print(f"  Questions skipped: {summary['skipped_total']}")
    print(f"    - declined     : {summary['skipped_declined']}")
    print(f"    - no context   : {summary['skipped_no_context']}")
    print(f"    - API errors   : {summary['skipped_api_error']}")
    if summary["low_faithfulness"]:
        print(f"\n  [WARNING] Low-faithfulness questions ({len(summary['low_faithfulness'])}):")
        for question in summary["low_faithfulness"][:5]:
            print(f"    - {question[:80]}...")
    print(f"\n  Full CSV  -> {out_csv}")
    print(f"  Summary   -> {out_json}")

    if gate_failures:
        print("\n  GATE STATUS: FAIL")
        for failure in gate_failures:
            print(f"    - {failure}")
        raise SystemExit(1)

    print("\n  GATE STATUS: PASS")


if __name__ == "__main__":
    main()
