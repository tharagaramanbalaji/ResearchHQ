from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ARXIV_MAX_RESULTS_PER_QUERY = 30_000
ARXIV_MIN_API_DELAY_SECONDS = 3.0


def run_step(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-paper corpus end to end.")
    parser.add_argument("--query", default=None, help='arXiv query, for example: "cat:cs.AI OR cat:cs.CL"')
    parser.add_argument("--from-date", default=None, help="Earliest arXiv submission date to include, YYYY-MM-DD.")
    parser.add_argument("--to-date", default=None, help="Latest arXiv submission date to include, YYYY-MM-DD.")
    parser.add_argument("--arxiv-ids-file", default=None, help="Text file with one arXiv ID or URL per line.")
    parser.add_argument("--openalex-search", default=None, help='OpenAlex search query, for example: "retrieval augmented generation"')
    parser.add_argument("--max-results", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=10.0)
    parser.add_argument("--pdf-sleep-seconds", type=float, default=3.0)
    parser.add_argument("--fetch-retries", type=int, default=6)
    parser.add_argument("--fetch-initial-wait-seconds", type=float, default=30.0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pdf-dir", action="append", default=[], help="PDF file or directory. Can be passed multiple times.")
    parser.add_argument(
        "--markdown-dir",
        action="append",
        default=[],
        help="Markdown file or directory. Can be passed multiple times.",
    )
    parser.add_argument("--web-urls-file", default=None, help="Text file containing one URL per line.")
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    parser.add_argument("--skip-download-pdfs", action="store_true")
    args = parser.parse_args()
    if (
        not args.query
        and not args.arxiv_ids_file
        and not args.openalex_search
        and not args.pdf_dir
        and not args.markdown_dir
        and not args.web_urls_file
    ):
        parser.error(
            "Provide at least one source: --query, --arxiv-ids-file, --openalex-search, --pdf-dir, --markdown-dir, or --web-urls-file"
        )
    if args.max_results < 1:
        parser.error("--max-results must be at least 1")
    if (args.query or args.arxiv_ids_file) and args.max_results > ARXIV_MAX_RESULTS_PER_QUERY:
        parser.error(f"--max-results cannot exceed {ARXIV_MAX_RESULTS_PER_QUERY:,} for one arXiv query")
    if (args.query or args.arxiv_ids_file) and args.sleep_seconds < ARXIV_MIN_API_DELAY_SECONDS:
        parser.error(f"--sleep-seconds must be at least {ARXIV_MIN_API_DELAY_SECONDS:.0f} seconds for arXiv API requests")
    return args


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    has_arxiv_documents = bool((args.query or args.arxiv_ids_file) and not args.skip_download_pdfs)
    has_openalex_documents = bool(args.openalex_search)

    if args.query or args.arxiv_ids_file:
        download_args = [
            str(script_dir / "download_arxiv.py"),
            "--max-results",
            str(args.max_results),
            "--sleep-seconds",
            str(args.sleep_seconds),
            "--pdf-sleep-seconds",
            str(args.pdf_sleep_seconds),
            "--fetch-retries",
            str(args.fetch_retries),
            "--fetch-initial-wait-seconds",
            str(args.fetch_initial_wait_seconds),
            "--data-dir",
            args.data_dir,
        ]
        if args.query:
            download_args.extend(["--query", args.query])
        if args.from_date:
            download_args.extend(["--from-date", args.from_date])
        if args.to_date:
            download_args.extend(["--to-date", args.to_date])
        if args.arxiv_ids_file:
            download_args.extend(["--arxiv-ids-file", args.arxiv_ids_file])
        if args.skip_download_pdfs:
            download_args.append("--metadata-only")

        run_step(download_args)

    if args.openalex_search:
        run_step(
            [
                str(script_dir / "download_openalex.py"),
                "--search",
                args.openalex_search,
                "--max-results",
                str(args.max_results),
                "--data-dir",
                args.data_dir,
                "--sleep-seconds",
                str(args.sleep_seconds),
                "--pdf-sleep-seconds",
                str(args.pdf_sleep_seconds),
            ]
        )

    if has_arxiv_documents or has_openalex_documents:
        run_step([str(script_dir / "extract_pdf_text.py"), "--data-dir", args.data_dir])

    if args.pdf_dir or args.markdown_dir or args.web_urls_file:
        ingest_args = [str(script_dir / "ingest_documents.py"), "--data-dir", args.data_dir]
        for pdf_dir in args.pdf_dir:
            ingest_args.extend(["--pdf-dir", pdf_dir])
        for markdown_dir in args.markdown_dir:
            ingest_args.extend(["--markdown-dir", markdown_dir])
        if args.web_urls_file:
            ingest_args.extend(["--web-urls-file", args.web_urls_file])
        if has_arxiv_documents or has_openalex_documents:
            ingest_args.append("--append")
        run_step(ingest_args)

    run_step(
        [
            str(script_dir / "chunk_corpus.py"),
            "--data-dir",
            args.data_dir,
            "--min-tokens",
            str(args.min_tokens),
            "--max-tokens",
            str(args.max_tokens),
            "--overlap-tokens",
            str(args.overlap_tokens),
        ]
    )


if __name__ == "__main__":
    main()
