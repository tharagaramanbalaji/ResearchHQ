#!/usr/bin/env python3
"""
corpus_metrics.py
=================
Comprehensive corpus health and metrics report.

Analyses documents, chunks, ChromaDB index, QA dataset, and latest Ragas
evaluation results.  Outputs a formatted console table, a machine-readable
JSON file, and a human-readable Markdown report.

Usage:
    python scripts/corpus_metrics.py --data-dir data/llm_ai_2023_2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from rag_corpus.paths import corpus_paths


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_stat(values: list[float | int]) -> dict:
    """Return min/max/mean/median/std for a numeric list, or zeros if empty."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def token_estimate(text: str) -> int:
    """Rough whitespace-based token count."""
    return len(text.split())


def histogram_buckets(values: list[int | float], n_buckets: int = 8) -> list[dict]:
    """Build a simple histogram of values into n_buckets bins."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"range": f"{lo}-{hi}", "count": len(values)}]
    width = math.ceil((hi - lo) / n_buckets)
    buckets = []
    for i in range(n_buckets):
        start = lo + i * width
        end = start + width
        count = sum(1 for v in values if start <= v < end) if i < n_buckets - 1 else sum(1 for v in values if start <= v <= end)
        buckets.append({"range": f"{start}-{end}", "count": count})
    return [b for b in buckets if b["count"] > 0]


# ── Document Metrics ─────────────────────────────────────────────────────────

def compute_document_metrics(documents_file: Path) -> dict:
    """Analyse documents.jsonl and return structured metrics."""
    docs = []
    if documents_file.exists():
        with documents_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))

    if not docs:
        return {"total": 0, "error": "documents.jsonl not found or empty"}

    source_types = Counter(d.get("source_type", "unknown") for d in docs)
    sources = Counter(d.get("source", "unknown") for d in docs)

    # Text lengths
    text_chars = [len(d.get("text", "")) for d in docs]
    text_tokens = [token_estimate(d.get("text", "")) for d in docs]

    # Temporal distribution
    years = Counter()
    for d in docs:
        pub = d.get("published_at", "")
        if pub:
            try:
                year = pub[:4]
                if year.isdigit():
                    years[int(year)] += 1
            except Exception:
                pass

    # Authors
    all_authors = []
    for d in docs:
        all_authors.extend(d.get("authors", []))
    author_counts = Counter(all_authors)
    unique_authors = len(author_counts)
    top_authors = author_counts.most_common(10)

    # arXiv categories
    categories = Counter()
    for d in docs:
        cats = d.get("categories", [])
        prim = d.get("primary_category")
        if prim:
            categories[prim] += 1
        for c in cats:
            if c != prim:
                categories[c] += 1

    # Page counts
    page_counts = [d.get("page_count", 0) for d in docs if d.get("page_count")]

    return {
        "total": len(docs),
        "source_type_distribution": dict(source_types.most_common()),
        "source_distribution": dict(sources.most_common()),
        "documents_per_year": dict(sorted(years.items())),
        "text_chars": safe_stat(text_chars),
        "text_tokens": safe_stat(text_tokens),
        "page_counts": safe_stat(page_counts),
        "unique_authors": unique_authors,
        "total_author_entries": len(all_authors),
        "top_authors": [{"name": name, "papers": count} for name, count in top_authors],
        "arxiv_categories": dict(categories.most_common(15)),
        "file_size_bytes": documents_file.stat().st_size if documents_file.exists() else 0,
    }


# ── Chunk Metrics ────────────────────────────────────────────────────────────

def compute_chunk_metrics(chunks_file: Path) -> dict:
    """Analyse chunks.jsonl and return structured metrics."""
    chunks = []
    if chunks_file.exists():
        with chunks_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

    if not chunks:
        return {"total": 0, "error": "chunks.jsonl not found or empty"}

    token_counts = [c.get("token_count_estimate", 0) for c in chunks]
    chunks_per_doc = Counter(c.get("document_id", "unknown") for c in chunks)
    cpd_values = list(chunks_per_doc.values())

    # Overlap verification: sample adjacent chunks from same document
    overlap_samples = []
    doc_chunks = {}
    for c in chunks:
        doc_id = c.get("document_id", "")
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append(c)

    for doc_id, dchunks in list(doc_chunks.items())[:10]:
        sorted_chunks = sorted(dchunks, key=lambda x: x.get("chunk_index", 0))
        for i in range(len(sorted_chunks) - 1):
            curr = sorted_chunks[i]
            nxt = sorted_chunks[i + 1]
            curr_tokens = set(curr.get("text", "").lower().split())
            nxt_tokens = set(nxt.get("text", "").lower().split())
            overlap = len(curr_tokens & nxt_tokens)
            overlap_samples.append(overlap)

    # Short chunks
    short_chunks = sum(1 for t in token_counts if t < 100)

    return {
        "total": len(chunks),
        "token_counts": safe_stat(token_counts),
        "token_histogram": histogram_buckets(token_counts),
        "chunks_per_document": safe_stat(cpd_values),
        "unique_documents": len(chunks_per_doc),
        "short_chunks_under_100": short_chunks,
        "overlap_samples": safe_stat(overlap_samples) if overlap_samples else {"note": "no adjacent chunks sampled"},
        "file_size_bytes": chunks_file.stat().st_size if chunks_file.exists() else 0,
    }


# ── ChromaDB Index Metrics ───────────────────────────────────────────────────

def compute_index_metrics(chroma_db_dir: Path, collection_name: str, total_chunks: int) -> dict:
    """Check ChromaDB collection state."""
    if not chroma_db_dir.exists():
        return {"status": "not_found", "error": f"ChromaDB directory not found at {chroma_db_dir}"}

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_db_dir))
        collection = client.get_collection(collection_name)
        count = collection.count()
        metadata = collection.metadata or {}

        drift = abs(count - total_chunks)
        drift_pct = (drift / total_chunks * 100) if total_chunks > 0 else 0

        return {
            "status": "ok",
            "collection_name": collection_name,
            "vector_count": count,
            "distance_metric": metadata.get("hnsw:space", "unknown"),
            "chunks_in_jsonl": total_chunks,
            "index_drift": drift,
            "index_drift_pct": round(drift_pct, 2),
            "in_sync": drift == 0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── QA Dataset Metrics ───────────────────────────────────────────────────────

def compute_qa_metrics(eval_dir: Path, total_documents: int) -> dict:
    """Analyse the QA dataset and latest Ragas results."""
    qa_json = eval_dir / "qa_dataset.json"
    ragas_json = eval_dir / "ragas_summary.json"

    result = {}

    if qa_json.exists():
        with qa_json.open("r", encoding="utf-8") as f:
            qa_entries = json.load(f)
        papers_covered = len(set(e.get("arxiv_id", e.get("title", "")) for e in qa_entries))
        coverage_pct = (papers_covered / total_documents * 100) if total_documents > 0 else 0
        result["qa_pairs"] = len(qa_entries)
        result["papers_covered"] = papers_covered
        result["corpus_coverage_pct"] = round(coverage_pct, 2)
    else:
        result["qa_pairs"] = 0
        result["qa_error"] = "qa_dataset.json not found"

    if ragas_json.exists():
        with ragas_json.open("r", encoding="utf-8") as f:
            ragas = json.load(f)
        result["ragas_faithfulness"] = ragas.get("faithfulness")
        result["ragas_answer_relevancy"] = ragas.get("answer_relevancy")
        result["ragas_n_evaluated"] = ragas.get("n_evaluated")
        result["ragas_skipped_total"] = ragas.get("skipped_total")
        low = ragas.get("low_faithfulness", [])
        result["ragas_low_faithfulness_count"] = len(low)
    else:
        result["ragas_error"] = "ragas_summary.json not found"

    return result


# ── Retrieval Latency Probe ──────────────────────────────────────────────────

def probe_retrieval_latency(chroma_db_dir: Path, collection_name: str) -> dict:
    """Time a handful of representative queries against the vector store."""
    if not chroma_db_dir.exists():
        return {"status": "skipped", "reason": "ChromaDB not found"}

    sample_queries = [
        "What is retrieval augmented generation?",
        "How do large language models handle long context?",
        "Explain the transformer architecture attention mechanism",
        "What are the challenges of fine-tuning LLMs?",
        "How is hallucination detected in language models?",
    ]

    try:
        from rag_corpus.retrieval import get_chroma_vectorstore
        vectorstore = get_chroma_vectorstore(chroma_db_dir, collection_name)

        latencies = []
        unique_docs = set()
        for query in sample_queries:
            start = time.time()
            results = vectorstore.similarity_search_with_score(query, k=5)
            elapsed = time.time() - start
            latencies.append(round(elapsed * 1000, 1))  # ms
            for doc, _ in results:
                unique_docs.add(doc.metadata.get("document_id", ""))

        return {
            "status": "ok",
            "queries_tested": len(sample_queries),
            "latency_ms": safe_stat(latencies),
            "unique_documents_in_results": len(unique_docs),
            "retrieval_diversity": round(len(unique_docs) / (len(sample_queries) * 5) * 100, 1),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Report Formatting ────────────────────────────────────────────────────────

def format_console_report(metrics: dict) -> str:
    """Format metrics as a readable console table."""
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  ENTERPRISE RAG CORPUS — METRICS REPORT")
    lines.append(f"  Generated: {datetime.now().isoformat()}")
    lines.append("=" * 72)

    # Documents
    dm = metrics.get("documents", {})
    lines.append("")
    lines.append("  📄 DOCUMENTS")
    lines.append("  " + "-" * 50)
    lines.append(f"  Total documents      : {dm.get('total', 0):,}")
    lines.append(f"  File size            : {dm.get('file_size_bytes', 0) / 1024 / 1024:.1f} MB")
    st = dm.get("source_type_distribution", {})
    if st:
        lines.append(f"  Source types         : {', '.join(f'{k}: {v}' for k, v in st.items())}")
    sd = dm.get("source_distribution", {})
    if sd:
        lines.append(f"  Source origins       : {', '.join(f'{k}: {v}' for k, v in sd.items())}")
    yrs = dm.get("documents_per_year", {})
    if yrs:
        lines.append(f"  By year              : {', '.join(f'{k}: {v}' for k, v in yrs.items())}")
    tt = dm.get("text_tokens", {})
    if tt.get("count"):
        lines.append(f"  Token range          : {tt['min']:,} – {tt['max']:,} (mean: {tt['mean']:,.0f}, median: {tt['median']:,.0f})")
    lines.append(f"  Unique authors       : {dm.get('unique_authors', 0):,}")
    cats = dm.get("arxiv_categories", {})
    if cats:
        top5 = list(cats.items())[:5]
        lines.append(f"  Top categories       : {', '.join(f'{k}: {v}' for k, v in top5)}")

    # Chunks
    cm = metrics.get("chunks", {})
    lines.append("")
    lines.append("  📦 CHUNKS")
    lines.append("  " + "-" * 50)
    lines.append(f"  Total chunks         : {cm.get('total', 0):,}")
    lines.append(f"  File size            : {cm.get('file_size_bytes', 0) / 1024 / 1024:.1f} MB")
    tc = cm.get("token_counts", {})
    if tc.get("count"):
        lines.append(f"  Token range          : {tc['min']} – {tc['max']} (mean: {tc['mean']:.0f}, median: {tc['median']:.0f})")
    cpd = cm.get("chunks_per_document", {})
    if cpd.get("count"):
        lines.append(f"  Chunks/doc           : {cpd['min']} – {cpd['max']} (mean: {cpd['mean']:.1f})")
    lines.append(f"  Short chunks (<100t) : {cm.get('short_chunks_under_100', 0)}")
    ov = cm.get("overlap_samples", {})
    if isinstance(ov, dict) and ov.get("count"):
        lines.append(f"  Overlap tokens (sampled): mean {ov['mean']:.0f}, median {ov['median']:.0f}")
    hist = cm.get("token_histogram", [])
    if hist:
        lines.append(f"  Token distribution   :")
        for b in hist:
            bar = "█" * min(b["count"] // 5, 40)
            lines.append(f"    {b['range']:>10s} │ {b['count']:>5d} {bar}")

    # Index
    im = metrics.get("index", {})
    lines.append("")
    lines.append("  🗄️  CHROMADB INDEX")
    lines.append("  " + "-" * 50)
    if im.get("status") == "ok":
        lines.append(f"  Vectors              : {im.get('vector_count', 0):,}")
        lines.append(f"  Distance metric      : {im.get('distance_metric', 'unknown')}")
        sync_icon = "✅" if im.get("in_sync") else "⚠️"
        lines.append(f"  In sync with chunks  : {sync_icon} (drift: {im.get('index_drift', 0)})")
    else:
        lines.append(f"  Status               : {im.get('status', 'unknown')} — {im.get('error', '')}")

    # Retrieval Latency
    rl = metrics.get("retrieval_latency", {})
    if rl.get("status") == "ok":
        lines.append("")
        lines.append("  ⚡ RETRIEVAL LATENCY (5 sample queries)")
        lines.append("  " + "-" * 50)
        lat = rl.get("latency_ms", {})
        lines.append(f"  Mean latency         : {lat.get('mean', 0):.0f} ms")
        lines.append(f"  Median latency       : {lat.get('median', 0):.0f} ms")
        lines.append(f"  Range                : {lat.get('min', 0):.0f} – {lat.get('max', 0):.0f} ms")
        lines.append(f"  Result diversity     : {rl.get('retrieval_diversity', 0):.0f}% unique docs")

    # QA & Evaluation
    qa = metrics.get("evaluation", {})
    lines.append("")
    lines.append("  📊 EVALUATION")
    lines.append("  " + "-" * 50)
    lines.append(f"  QA pairs             : {qa.get('qa_pairs', 0)}")
    lines.append(f"  Papers covered       : {qa.get('papers_covered', 0)}")
    lines.append(f"  Corpus coverage      : {qa.get('corpus_coverage_pct', 0):.1f}%")
    if qa.get("ragas_faithfulness") is not None:
        lines.append(f"  Faithfulness (Ragas) : {qa['ragas_faithfulness']:.3f}")
        lines.append(f"  Answer Relevancy     : {qa['ragas_answer_relevancy']:.3f}")
        lines.append(f"  Evaluated questions  : {qa.get('ragas_n_evaluated', 0)}")
        lines.append(f"  Low faithfulness     : {qa.get('ragas_low_faithfulness_count', 0)}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def format_markdown_report(metrics: dict) -> str:
    """Format metrics as a Markdown document."""
    dm = metrics.get("documents", {})
    cm = metrics.get("chunks", {})
    im = metrics.get("index", {})
    qa = metrics.get("evaluation", {})
    rl = metrics.get("retrieval_latency", {})

    md = []
    md.append("# Enterprise RAG Corpus — Metrics Report")
    md.append("")
    md.append(f"**Generated:** {metrics.get('generated_at', 'unknown')}  ")
    md.append(f"**Data Directory:** `{metrics.get('data_dir', 'unknown')}`")
    md.append("")
    md.append("---")
    md.append("")

    # Summary Table
    md.append("## Summary")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|:---|:---|")
    md.append(f"| **Documents** | {dm.get('total', 0):,} |")
    md.append(f"| **Chunks** | {cm.get('total', 0):,} |")
    vc = im.get("vector_count", "N/A")
    md.append(f"| **ChromaDB Vectors** | {vc:,} |" if isinstance(vc, int) else f"| **ChromaDB Vectors** | {vc} |")
    md.append(f"| **QA Pairs** | {qa.get('qa_pairs', 0)} |")
    if qa.get("ragas_faithfulness") is not None:
        md.append(f"| **Faithfulness** | {qa['ragas_faithfulness']:.3f} |")
        md.append(f"| **Answer Relevancy** | {qa['ragas_answer_relevancy']:.3f} |")
    md.append("")

    # Documents
    md.append("## Documents")
    md.append("")
    st = dm.get("source_type_distribution", {})
    if st:
        md.append("### Source Type Distribution")
        md.append("")
        md.append("| Type | Count |")
        md.append("|:---|---:|")
        for k, v in st.items():
            md.append(f"| {k} | {v} |")
        md.append("")

    sd = dm.get("source_distribution", {})
    if sd:
        md.append("### Source Origin Distribution")
        md.append("")
        md.append("| Source | Count |")
        md.append("|:---|---:|")
        for k, v in sd.items():
            md.append(f"| {k} | {v} |")
        md.append("")

    yrs = dm.get("documents_per_year", {})
    if yrs:
        md.append("### Documents per Year")
        md.append("")
        md.append("| Year | Count |")
        md.append("|:---|---:|")
        for k, v in sorted(yrs.items()):
            md.append(f"| {k} | {v} |")
        md.append("")

    tt = dm.get("text_tokens", {})
    if tt.get("count"):
        md.append("### Document Text Statistics")
        md.append("")
        md.append("| Stat | Tokens | Characters |")
        md.append("|:---|---:|---:|")
        tc = dm.get("text_chars", {})
        md.append(f"| Min | {tt['min']:,} | {tc.get('min', 0):,} |")
        md.append(f"| Max | {tt['max']:,} | {tc.get('max', 0):,} |")
        md.append(f"| Mean | {tt['mean']:,.0f} | {tc.get('mean', 0):,.0f} |")
        md.append(f"| Median | {tt['median']:,.0f} | {tc.get('median', 0):,.0f} |")
        md.append(f"| Std Dev | {tt['std']:,.0f} | {tc.get('std', 0):,.0f} |")
        md.append("")

    cats = dm.get("arxiv_categories", {})
    if cats:
        md.append("### arXiv Category Distribution")
        md.append("")
        md.append("| Category | Count |")
        md.append("|:---|---:|")
        for k, v in cats.items():
            md.append(f"| `{k}` | {v} |")
        md.append("")

    top_a = dm.get("top_authors", [])
    if top_a:
        md.append("### Top Authors")
        md.append("")
        md.append("| Author | Papers |")
        md.append("|:---|---:|")
        for a in top_a[:10]:
            md.append(f"| {a['name']} | {a['papers']} |")
        md.append("")

    # Chunks
    md.append("## Chunks")
    md.append("")
    tk = cm.get("token_counts", {})
    cpd = cm.get("chunks_per_document", {})
    md.append("| Metric | Value |")
    md.append("|:---|:---|")
    md.append(f"| Total Chunks | {cm.get('total', 0):,} |")
    if tk.get("count"):
        md.append(f"| Token Range | {tk['min']} – {tk['max']} |")
        md.append(f"| Mean Tokens | {tk['mean']:.0f} |")
    if cpd.get("count"):
        md.append(f"| Chunks/Doc Range | {cpd['min']} – {cpd['max']} |")
        md.append(f"| Mean Chunks/Doc | {cpd['mean']:.1f} |")
    md.append(f"| Short Chunks (<100 tokens) | {cm.get('short_chunks_under_100', 0)} |")
    md.append("")

    hist = cm.get("token_histogram", [])
    if hist:
        md.append("### Token Count Distribution")
        md.append("")
        md.append("| Token Range | Count |")
        md.append("|:---|---:|")
        for b in hist:
            md.append(f"| {b['range']} | {b['count']} |")
        md.append("")

    # Index
    md.append("## ChromaDB Index")
    md.append("")
    if im.get("status") == "ok":
        sync = "✅ Yes" if im.get("in_sync") else f"⚠️ No (drift: {im.get('index_drift', 0)})"
        md.append("| Metric | Value |")
        md.append("|:---|:---|")
        md.append(f"| Vectors | {im.get('vector_count', 0):,} |")
        md.append(f"| Distance Metric | {im.get('distance_metric', 'unknown')} |")
        md.append(f"| In Sync | {sync} |")
    else:
        md.append(f"> ⚠️ Index status: {im.get('error', 'unknown')}")
    md.append("")

    # Latency
    if rl.get("status") == "ok":
        md.append("## Retrieval Latency")
        md.append("")
        lat = rl.get("latency_ms", {})
        md.append("| Metric | Value |")
        md.append("|:---|:---|")
        md.append(f"| Queries Tested | {rl.get('queries_tested', 0)} |")
        md.append(f"| Mean Latency | {lat.get('mean', 0):.0f} ms |")
        md.append(f"| Median Latency | {lat.get('median', 0):.0f} ms |")
        md.append(f"| Range | {lat.get('min', 0):.0f} – {lat.get('max', 0):.0f} ms |")
        md.append(f"| Result Diversity | {rl.get('retrieval_diversity', 0):.0f}% unique docs |")
        md.append("")

    # Evaluation
    md.append("## Evaluation")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|:---|:---|")
    md.append(f"| QA Pairs | {qa.get('qa_pairs', 0)} |")
    md.append(f"| Papers Covered | {qa.get('papers_covered', 0)} |")
    md.append(f"| Corpus Coverage | {qa.get('corpus_coverage_pct', 0):.1f}% |")
    if qa.get("ragas_faithfulness") is not None:
        md.append(f"| Faithfulness | {qa['ragas_faithfulness']:.3f} |")
        md.append(f"| Answer Relevancy | {qa['ragas_answer_relevancy']:.3f} |")
        md.append(f"| Evaluated | {qa.get('ragas_n_evaluated', 0)} |")
        md.append(f"| Low Faithfulness | {qa.get('ragas_low_faithfulness_count', 0)} |")
    md.append("")

    md.append("---")
    md.append("*Generated by `scripts/corpus_metrics.py`*")
    return "\n".join(md)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and report corpus metrics.")
    parser.add_argument("--data-dir", default="data/llm_ai_2023_2026")
    parser.add_argument("--collection-name", default="research_papers")
    parser.add_argument("--skip-latency", action="store_true", help="Skip retrieval latency probe (requires model loading)")
    parser.add_argument("--output-dir", default=None, help="Directory for output files (default: data/evaluation)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = corpus_paths(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing document metrics...", flush=True)
    doc_metrics = compute_document_metrics(paths.documents_jsonl)

    print("Computing chunk metrics...", flush=True)
    chunk_metrics = compute_chunk_metrics(paths.chunks_jsonl)

    print("Computing index metrics...", flush=True)
    chroma_db_dir = paths.processed_dir / "chroma_db"
    index_metrics = compute_index_metrics(chroma_db_dir, args.collection_name, chunk_metrics.get("total", 0))

    retrieval_metrics = {}
    if not args.skip_latency and index_metrics.get("status") == "ok":
        print("Probing retrieval latency (loading BGE model)...", flush=True)
        retrieval_metrics = probe_retrieval_latency(chroma_db_dir, args.collection_name)
    else:
        retrieval_metrics = {"status": "skipped"}

    print("Computing evaluation metrics...", flush=True)
    eval_dir = PROJECT_ROOT / "data" / "evaluation"
    qa_metrics = compute_qa_metrics(eval_dir, doc_metrics.get("total", 0))

    all_metrics = {
        "generated_at": datetime.now().isoformat(),
        "data_dir": args.data_dir,
        "documents": doc_metrics,
        "chunks": chunk_metrics,
        "index": index_metrics,
        "retrieval_latency": retrieval_metrics,
        "evaluation": qa_metrics,
    }

    # Write JSON
    json_path = output_dir / "corpus_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # Write Markdown
    md_path = output_dir / "corpus_metrics.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(format_markdown_report(all_metrics))

    # Console
    print(format_console_report(all_metrics))
    print(f"\n  Reports saved:")
    print(f"    JSON → {json_path}")
    print(f"    MD   → {md_path}")


if __name__ == "__main__":
    main()
