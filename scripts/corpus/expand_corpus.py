#!/usr/bin/env python3
"""
expand_corpus.py
================
Orchestrates multi-wave corpus expansion to enterprise scale (500+ documents).

Runs arXiv downloads across multiple research-area queries, OpenAlex scholarly
paper searches, and web page ingestion — then re-chunks and re-indexes the
full corpus.

Usage:
    .\\venv\\Scripts\\python.exe scripts/corpus/expand_corpus.py --data-dir data/llm_ai_2023_2026

    # Dry run (show what would happen without downloading):
    .\\venv\\Scripts\\python.exe scripts/corpus/expand_corpus.py --data-dir data/llm_ai_2023_2026 --dry-run

    # Skip specific waves:
    .\\venv\\Scripts\\python.exe scripts/corpus/expand_corpus.py --data-dir data/llm_ai_2023_2026 --skip-arxiv --skip-openalex
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

# arXiv date range for recent papers
ARXIV_FROM_DATE = "2023-06-03"
ARXIV_TO_DATE = "2026-06-12"

# ── Wave Definitions ─────────────────────────────────────────────────────────

ARXIV_WAVES = [
    {
        "name": "RAG & Information Retrieval",
        "query": '((cat:cs.IR OR cat:cs.CL) AND (all:"retrieval augmented generation" OR all:RAG OR (all:retrieval AND all:augmented)))',
        "max_results": 50,
    },
    {
        "name": "Reasoning & Agents",
        "query": '((cat:cs.AI OR cat:cs.MA OR cat:cs.CL) AND (all:"chain of thought" OR all:reasoning OR all:"language agent" OR all:"tool use"))',
        "max_results": 50,
    },
    {
        "name": "Fine-tuning & Alignment",
        "query": '((cat:cs.LG OR cat:cs.CL) AND (all:RLHF OR all:"instruction tuning" OR all:alignment OR all:"preference optimization" OR all:DPO))',
        "max_results": 50,
    },
    {
        "name": "Multimodal & Vision-Language",
        "query": '((cat:cs.CV OR cat:cs.CL) AND (all:"vision language" OR all:multimodal OR all:VLM OR all:"visual question answering"))',
        "max_results": 50,
    },
    {
        "name": "Safety, Evaluation & Benchmarks",
        "query": '((cat:cs.AI OR cat:cs.CL OR cat:cs.CR) AND (all:"LLM evaluation" OR all:benchmark OR all:hallucination OR all:"red teaming" OR all:safety))',
        "max_results": 50,
    },
]

OPENALEX_SEARCHES = [
    {"search": "retrieval augmented generation", "max_results": 25},
    {"search": "large language model reasoning", "max_results": 25},
    {"search": "instruction tuning alignment RLHF", "max_results": 25},
    {"search": "transformer architecture efficiency", "max_results": 25},
]


def count_jsonl(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_pdfs(papers_dir: Path) -> int:
    """Count PDFs in the papers directory."""
    if not papers_dir.exists():
        return 0
    return sum(1 for f in papers_dir.glob("*.pdf"))


def run_step(args: list[str], dry_run: bool = False) -> bool:
    """Run a subprocess, returning True on success."""
    cmd = [PYTHON] + args
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd[:4])}...")
        return True
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  [WARNING] Step failed with exit code {e.returncode}", flush=True)
        return False
    except Exception as e:
        print(f"  [ERROR] {e}", flush=True)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand corpus to enterprise scale.")
    parser.add_argument("--data-dir", default="data/llm_ai_2023_2026")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--skip-arxiv", action="store_true", help="Skip arXiv download waves")
    parser.add_argument("--skip-openalex", action="store_true", help="Skip OpenAlex download waves")
    parser.add_argument("--skip-web", action="store_true", help="Skip web page ingestion")
    parser.add_argument("--skip-rechunk", action="store_true", help="Skip re-chunking step")
    parser.add_argument("--skip-reindex", action="store_true", help="Skip re-indexing step")
    parser.add_argument("--sleep-seconds", type=float, default=10.0, help="Delay between arXiv API requests")
    parser.add_argument("--pdf-sleep-seconds", type=float, default=5.0, help="Delay between PDF downloads")
    parser.add_argument("--wave-delay", type=float, default=30.0, help="Delay between arXiv waves (seconds)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    # Paths for counting
    sys.path.append(str(PROJECT_ROOT / "src"))
    from rag_corpus.paths import corpus_paths
    paths = corpus_paths(data_dir)

    # ── Before snapshot ──────────────────────────────────────────────────────
    before_docs = count_jsonl(paths.documents_jsonl)
    before_chunks = count_jsonl(paths.chunks_jsonl)
    before_pdfs = count_pdfs(paths.papers_dir)

    print("=" * 65)
    print("  ENTERPRISE CORPUS EXPANSION")
    print("=" * 65)
    print(f"  Data directory  : {data_dir}")
    print(f"  Before — PDFs   : {before_pdfs}")
    print(f"  Before — Docs   : {before_docs}")
    print(f"  Before — Chunks : {before_chunks}")
    if args.dry_run:
        print("  Mode            : DRY RUN")
    print("=" * 65)

    wave_num = 0

    # ── Wave 1: arXiv ────────────────────────────────────────────────────────
    if not args.skip_arxiv:
        print(f"\n{'─' * 65}")
        print("  WAVE 1: arXiv Expansion")
        print(f"{'─' * 65}")
        for i, wave in enumerate(ARXIV_WAVES, 1):
            wave_num += 1
            print(f"\n  [{wave_num}] {wave['name']} (target: {wave['max_results']} papers)")

            download_args = [
                str(SCRIPT_DIR / "download_arxiv.py"),
                "--query", wave["query"],
                "--from-date", ARXIV_FROM_DATE,
                "--to-date", ARXIV_TO_DATE,
                "--max-results", str(wave["max_results"]),
                "--sleep-seconds", str(args.sleep_seconds),
                "--pdf-sleep-seconds", str(args.pdf_sleep_seconds),
                "--data-dir", data_dir,
            ]
            run_step(download_args, dry_run=args.dry_run)

            if i < len(ARXIV_WAVES) and not args.dry_run:
                print(f"  Waiting {args.wave_delay:.0f}s between arXiv waves...", flush=True)
                time.sleep(args.wave_delay)

        # Extract text from all new PDFs
        if not args.dry_run:
            print(f"\n  Extracting text from all PDFs...")
            run_step([str(SCRIPT_DIR / "extract_pdf_text.py"), "--data-dir", data_dir])

    # ── Wave 2: OpenAlex ─────────────────────────────────────────────────────
    if not args.skip_openalex:
        print(f"\n{'─' * 65}")
        print("  WAVE 2: OpenAlex Expansion")
        print(f"{'─' * 65}")
        for search_config in OPENALEX_SEARCHES:
            wave_num += 1
            print(f"\n  [{wave_num}] Search: \"{search_config['search']}\" (target: {search_config['max_results']})")

            openalex_args = [
                str(SCRIPT_DIR / "download_openalex.py"),
                "--search", search_config["search"],
                "--max-results", str(search_config["max_results"]),
                "--data-dir", data_dir,
            ]
            run_step(openalex_args, dry_run=args.dry_run)

        # Extract text from OpenAlex PDFs too
        if not args.dry_run:
            print(f"\n  Extracting text from all PDFs...")
            run_step([str(SCRIPT_DIR / "extract_pdf_text.py"), "--data-dir", data_dir])

    # ── Wave 3: Web Pages ────────────────────────────────────────────────────
    if not args.skip_web:
        print(f"\n{'─' * 65}")
        print("  WAVE 3: Web Page Ingestion")
        print(f"{'─' * 65}")
        web_urls_file = paths.raw_dir / "web_urls_enterprise.txt"
        if web_urls_file.exists():
            wave_num += 1
            print(f"\n  [{wave_num}] Ingesting URLs from {web_urls_file}")
            ingest_args = [
                str(SCRIPT_DIR / "ingest_documents.py"),
                "--data-dir", data_dir,
                "--web-urls-file", str(web_urls_file),
                "--append",
            ]
            run_step(ingest_args, dry_run=args.dry_run)
        else:
            print(f"  Skipped — no web_urls_enterprise.txt found at {web_urls_file}")

    # ── Re-chunk ─────────────────────────────────────────────────────────────
    if not args.skip_rechunk:
        print(f"\n{'─' * 65}")
        print("  RE-CHUNKING FULL CORPUS")
        print(f"{'─' * 65}")
        chunk_args = [
            str(SCRIPT_DIR / "chunk_corpus.py"),
            "--data-dir", data_dir,
            "--min-tokens", "500",
            "--max-tokens", "800",
            "--overlap-tokens", "100",
        ]
        run_step(chunk_args, dry_run=args.dry_run)

    # ── Re-index ─────────────────────────────────────────────────────────────
    if not args.skip_reindex:
        print(f"\n{'─' * 65}")
        print("  RE-INDEXING CHROMADB")
        print(f"{'─' * 65}")
        index_args = [
            str(SCRIPT_DIR / "index_chroma.py"),
            "--data-dir", data_dir,
        ]
        run_step(index_args, dry_run=args.dry_run)

    # ── After snapshot ───────────────────────────────────────────────────────
    after_docs = count_jsonl(paths.documents_jsonl)
    after_chunks = count_jsonl(paths.chunks_jsonl)
    after_pdfs = count_pdfs(paths.papers_dir)

    print(f"\n{'=' * 65}")
    print("  EXPANSION COMPLETE")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<20s} {'Before':>10s} {'After':>10s} {'Delta':>10s}")
    print(f"  {'─' * 52}")
    print(f"  {'PDFs':<20s} {before_pdfs:>10,} {after_pdfs:>10,} {after_pdfs - before_pdfs:>+10,}")
    print(f"  {'Documents':<20s} {before_docs:>10,} {after_docs:>10,} {after_docs - before_docs:>+10,}")
    print(f"  {'Chunks':<20s} {before_chunks:>10,} {after_chunks:>10,} {after_chunks - before_chunks:>+10,}")
    print(f"{'=' * 65}")

    if args.dry_run:
        print("\n  This was a DRY RUN. No downloads or changes were made.")
        print("  Remove --dry-run to execute for real.")


if __name__ == "__main__":
    main()
