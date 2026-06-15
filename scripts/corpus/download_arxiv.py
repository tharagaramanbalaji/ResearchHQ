from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
import sys

import arxiv

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from rag_corpus.files import sha256_file
from rag_corpus.manifest import write_manifest
from rag_corpus.paths import corpus_paths


ARXIV_MAX_RESULTS_PER_QUERY = 30_000
ARXIV_MIN_API_DELAY_SECONDS = 3.0
ARXIV_DATE_FORMAT = "%Y-%m-%d"


def safe_arxiv_id(raw_id: str) -> str:
    paper_id = raw_id.rstrip("/").split("/")[-1]
    return "arxiv_" + paper_id.replace("/", "_")


def normalize_arxiv_id(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.rstrip("/").split("/")[-1]
    if cleaned.endswith(".pdf"):
        cleaned = cleaned[:-4]
    return cleaned


def read_arxiv_ids(path: Path) -> list[str]:
    ids = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        paper_id = normalize_arxiv_id(stripped)
        if paper_id and paper_id not in seen:
            ids.append(paper_id)
            seen.add(paper_id)
    return ids


def parse_date(value: str, option_name: str) -> datetime:
    try:
        return datetime.strptime(value, ARXIV_DATE_FORMAT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option_name} must use YYYY-MM-DD format") from exc


def arxiv_date(value: datetime, end_of_day: bool = False) -> str:
    suffix = "2359" if end_of_day else "0000"
    return value.strftime("%Y%m%d") + suffix


def with_submitted_date_filter(query: str, from_date: str | None, to_date: str | None) -> str:
    if not from_date and not to_date:
        return query

    start = parse_date(from_date, "--from-date") if from_date else datetime.strptime("1991-01-01", ARXIV_DATE_FORMAT)
    end = parse_date(to_date, "--to-date") if to_date else datetime.utcnow()
    if start > end:
        raise argparse.ArgumentTypeError("--from-date must be earlier than or equal to --to-date")

    date_filter = f"submittedDate:[{arxiv_date(start)} TO {arxiv_date(end, end_of_day=True)}]"
    return f"({query}) AND {date_filter}"


def retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    retry_after = error.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def download_file(url: str, destination: Path, retries: int = 5, initial_wait_seconds: float = 15.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "enterprise-rag-corpus/0.1 (research corpus builder)"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                destination.write_bytes(response.read())
            return
        except urllib.error.HTTPError as exc:
            if attempt == retries:
                raise
            wait_seconds = retry_after_seconds(exc) or initial_wait_seconds * attempt
            print(f"Download was rate limited or blocked; waiting {wait_seconds:.0f}s before retry {attempt + 1}/{retries}")
            time.sleep(wait_seconds)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(initial_wait_seconds * attempt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download arXiv paper metadata and PDFs.")
    parser.add_argument("--query", default=None, help='arXiv query, for example: "cat:cs.AI OR cat:cs.CL"')
    parser.add_argument("--from-date", default=None, help="Earliest arXiv submission date to include, YYYY-MM-DD.")
    parser.add_argument("--to-date", default=None, help="Latest arXiv submission date to include, YYYY-MM-DD.")
    parser.add_argument(
        "--arxiv-ids-file",
        default=None,
        help="Text file with one arXiv ID or arXiv URL per line.",
    )
    parser.add_argument("--max-results", type=int, default=25)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=10.0,
        help="Delay between arXiv API requests. Values below 3 seconds are rejected.",
    )
    parser.add_argument("--pdf-sleep-seconds", type=float, default=3.0, help="Delay between PDF downloads.")
    parser.add_argument("--fetch-retries", type=int, default=6)
    parser.add_argument("--fetch-initial-wait-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if not args.query and not args.arxiv_ids_file:
        parser.error("Provide --query or --arxiv-ids-file")
    if args.arxiv_ids_file and (args.from_date or args.to_date):
        parser.error("--from-date and --to-date are only supported with --query, not --arxiv-ids-file")
    if args.query and (args.from_date or args.to_date):
        try:
            args.query = with_submitted_date_filter(args.query, args.from_date, args.to_date)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    if args.max_results < 1:
        parser.error("--max-results must be at least 1")
    if args.max_results > ARXIV_MAX_RESULTS_PER_QUERY:
        parser.error(f"--max-results cannot exceed {ARXIV_MAX_RESULTS_PER_QUERY:,} for one arXiv query")
    if args.sleep_seconds < ARXIV_MIN_API_DELAY_SECONDS:
        parser.error(f"--sleep-seconds must be at least {ARXIV_MIN_API_DELAY_SECONDS:.0f} seconds for arXiv API requests")
    return args


def main() -> None:
    args = parse_args()
    paths = corpus_paths(args.data_dir)
    paths.ensure()

    downloaded = 0
    skipped_existing = 0
    metadata_written = 0
    failures: list[dict[str, str]] = []

    explicit_ids = read_arxiv_ids(Path(args.arxiv_ids_file)) if args.arxiv_ids_file else []

    # Configure arXiv client with delay_seconds and retries matching args
    client = arxiv.Client(
        delay_seconds=args.sleep_seconds,
        num_retries=args.fetch_retries
    )

    if explicit_ids:
        total_requested = min(args.max_results, len(explicit_ids))
        target_ids = explicit_ids[:total_requested]
        search = arxiv.Search(id_list=target_ids, max_results=total_requested)
        target_new = total_requested
    else:
        # Search up to the query maximum limit, but stop fetching once target_new new downloads are completed
        search = arxiv.Search(
            query=args.query,
            max_results=ARXIV_MAX_RESULTS_PER_QUERY,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        target_new = args.max_results

    print(f"Fetching arXiv records using 'arxiv' library (target: {target_new} new downloads)...", flush=True)

    new_downloads_count = 0
    try:
        results_generator = client.results(search)
        for result in results_generator:
            if new_downloads_count >= target_new:
                break

            doc_id = safe_arxiv_id(result.entry_id)
            metadata_path = paths.metadata_dir / f"{doc_id}.json"
            pdf_path = paths.papers_dir / f"{doc_id}.pdf"

            pdf_exists = pdf_path.exists()
            metadata_exists = metadata_path.exists()

            # Skip processing if we already have the target file(s)
            if args.metadata_only:
                if metadata_exists:
                    skipped_existing += 1
                    print(f"Metadata already exists for {doc_id} (skipping)", flush=True)
                    continue
            else:
                if pdf_exists:
                    skipped_existing += 1
                    print(f"PDF already exists for {doc_id} (skipping)", flush=True)
                    # If metadata JSON doesn't exist, generate it for consistency
                    if not metadata_exists:
                        external_id = result.get_short_id()
                        html_url = result.entry_id
                        for link in result.links:
                            if link.rel == "alternate":
                                html_url = link.href
                                break
                        record = {
                            "document_id": doc_id,
                            "source": "arxiv",
                            "source_type": "pdf",
                            "external_id": external_id,
                            "url": html_url,
                            "pdf_url": result.pdf_url,
                            "title": normalize_space(result.title),
                            "abstract": normalize_space(result.summary),
                            "authors": [normalize_space(author.name) for author in result.authors],
                            "published_at": result.published.isoformat().replace("+00:00", "Z"),
                            "updated_at": result.updated.isoformat().replace("+00:00", "Z"),
                            "categories": result.categories,
                            "primary_category": result.primary_category,
                            "metadata_path": str(metadata_path),
                            "pdf_path": str(pdf_path),
                            "sha256": sha256_file(pdf_path)
                        }
                        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                        metadata_written += 1
                    continue

            external_id = result.get_short_id()

            # Find the html_url (alternate link)
            html_url = result.entry_id
            for link in result.links:
                if link.rel == "alternate":
                    html_url = link.href
                    break

            record = {
                "document_id": doc_id,
                "source": "arxiv",
                "source_type": "pdf",
                "external_id": external_id,
                "url": html_url,
                "pdf_url": result.pdf_url,
                "title": normalize_space(result.title),
                "abstract": normalize_space(result.summary),
                "authors": [normalize_space(author.name) for author in result.authors],
                "published_at": result.published.isoformat().replace("+00:00", "Z"),
                "updated_at": result.updated.isoformat().replace("+00:00", "Z"),
                "categories": result.categories,
                "primary_category": result.primary_category,
            }

            record["metadata_path"] = str(metadata_path)
            if record.get("pdf_url"):
                record["pdf_path"] = str(pdf_path)

            if not args.metadata_only and record.get("pdf_url"):
                try:
                    print(
                        f"PDF {new_downloads_count + 1}/{target_new}: downloading {doc_id}",
                        flush=True,
                    )
                    download_file(
                        record["pdf_url"],
                        pdf_path,
                        retries=args.fetch_retries,
                        initial_wait_seconds=args.fetch_initial_wait_seconds
                    )
                    downloaded += 1
                    print(
                        f"PDF {new_downloads_count + 1}/{target_new}: saved {pdf_path}",
                        flush=True,
                    )
                    record["sha256"] = sha256_file(pdf_path)
                    time.sleep(args.pdf_sleep_seconds)
                except Exception as exc:
                    failures.append({"document_id": doc_id, "error": str(exc)})
                    record["download_error"] = str(exc)

            metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            metadata_written += 1
            new_downloads_count += 1

    except (arxiv.HTTPError, urllib.error.HTTPError) as exc:
        status = getattr(exc, "status", getattr(exc, "code", None))
        if status == 403:
            print(
                "\nERROR: HTTP 403 Forbidden\n"
                "Means your IP has been explicitly flagged for a temporary block due to aggressive harvesting.\n"
                "Aborting request to prevent extending the ban.\n",
                file=sys.stderr,
                flush=True,
            )
        elif status == 429:
            print(
                "\nERROR: HTTP 429 Too Many Requests\n"
                "The arXiv API rate limited this request. Your IP might be temporarily blocked.\n",
                file=sys.stderr,
                flush=True,
            )
        elif status == 503:
            print(
                "\nERROR: HTTP 503 Service Unavailable\n"
                "Means the server capacity is temporarily exceeded or your program is making requests too quickly.\n",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"Error fetching results from arXiv API: {exc}", file=sys.stderr, flush=True)
        if metadata_written == 0:
            raise
    except Exception as exc:
        print(f"Error fetching results from arXiv API: {exc}", file=sys.stderr, flush=True)
        if metadata_written == 0:
            raise

    manifest_path = write_manifest(
        paths,
        step="download_arxiv",
        parameters={
            "query": args.query,
            "from_date": args.from_date,
            "to_date": args.to_date,
            "arxiv_ids_file": args.arxiv_ids_file,
            "max_results": args.max_results,
            "metadata_only": args.metadata_only,
            "sleep_seconds": args.sleep_seconds,
            "pdf_sleep_seconds": args.pdf_sleep_seconds,
            "fetch_retries": args.fetch_retries,
            "fetch_initial_wait_seconds": args.fetch_initial_wait_seconds,
        },
        metrics={
            "metadata_written": metadata_written,
            "pdfs_downloaded": downloaded,
            "pdfs_skipped_existing": skipped_existing,
            "failures": len(failures),
        },
        failures=failures,
    )
    print(f"Wrote {metadata_written} metadata records")
    print(f"Downloaded {downloaded} PDFs; skipped {skipped_existing} existing PDFs")
    if failures:
        print(f"Failures: {len(failures)}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
