from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from rag_corpus.files import sha256_file
from rag_corpus.manifest import write_manifest
from rag_corpus.paths import corpus_paths


OPENALEX_API = "https://api.openalex.org/works"


def request_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "enterprise-rag-corpus/0.1 (mailto:research@example.com)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_works(search: str, max_results: int, per_page: int, sleep_seconds: float) -> list[dict]:
    works: list[dict] = []
    cursor = "*"

    while len(works) < max_results:
        params = urllib.parse.urlencode(
            {
                "search": search,
                "filter": "open_access.is_oa:true",
                "per-page": min(per_page, max_results - len(works)),
                "cursor": cursor,
                "sort": "cited_by_count:desc",
            }
        )
        payload = request_json(f"{OPENALEX_API}?{params}")
        results = payload.get("results", [])
        if not results:
            break
        works.extend(results)
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(sleep_seconds)

    return works[:max_results]


def inverted_index_to_text(index: dict | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, offsets in index.items():
        for offset in offsets:
            positions.append((offset, word))
    return " ".join(word for _, word in sorted(positions))


def best_pdf_url(work: dict) -> str | None:
    primary = (work.get("open_access") or {}).get("oa_url")
    candidates = [primary]
    for location in work.get("locations", []) or []:
        candidates.append((location or {}).get("pdf_url"))
        candidates.append(((location or {}).get("source") or {}).get("homepage_url"))

    for candidate in candidates:
        if candidate and ".pdf" in candidate.lower():
            return candidate
    return primary


def download_pdf(url: str, destination: Path, retries: int = 3, sleep_seconds: float = 5.0) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "enterprise-rag-corpus/0.1 (mailto:research@example.com)"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                content_type = response.headers.get("content-type", "").lower()
                body = response.read()
            if "pdf" not in content_type and not body.startswith(b"%PDF"):
                return False
            destination.write_bytes(body)
            return True
        except Exception:
            if attempt == retries:
                return False
            time.sleep(sleep_seconds * attempt)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download open-access papers discovered through OpenAlex.")
    parser.add_argument("--search", required=True, help='Search query, for example: "large language models retrieval"')
    parser.add_argument("--max-results", type=int, default=25)
    parser.add_argument("--per-page", type=int, default=25)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--pdf-sleep-seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = corpus_paths(args.data_dir)
    paths.ensure()

    works = iter_works(args.search, args.max_results, args.per_page, args.sleep_seconds)
    metadata_written = 0
    downloaded = 0
    skipped = 0
    failures: list[dict[str, str]] = []

    total_requested = len(works)
    for index, work in enumerate(works, start=1):
        openalex_id = str(work.get("id", "")).rstrip("/").split("/")[-1]
        document_id = f"openalex_{openalex_id}"
        metadata_path = paths.metadata_dir / f"{document_id}.json"
        pdf_path = paths.papers_dir / f"{document_id}.pdf"
        pdf_url = best_pdf_url(work)

        record = {
            "document_id": document_id,
            "source": "openalex",
            "source_type": "pdf",
            "external_id": openalex_id,
            "url": work.get("id"),
            "pdf_url": pdf_url,
            "title": work.get("title") or "",
            "authors": [
                ((author.get("author") or {}).get("display_name") or "")
                for author in work.get("authorships", [])
                if author.get("author")
            ],
            "abstract": inverted_index_to_text(work.get("abstract_inverted_index")),
            "published_at": str(work.get("publication_date") or ""),
            "updated_at": str(work.get("updated_date") or ""),
            "metadata_path": str(metadata_path),
            "pdf_path": str(pdf_path),
            "open_access": work.get("open_access"),
        }

        if pdf_url:
            if pdf_path.exists():
                record["sha256"] = sha256_file(pdf_path)
                skipped += 1
                print(f"PDF {index}/{total_requested}: already exists for {document_id}", flush=True)
            elif download_pdf(pdf_url, pdf_path, sleep_seconds=args.pdf_sleep_seconds):
                record["sha256"] = sha256_file(pdf_path)
                downloaded += 1
                print(f"PDF {index}/{total_requested}: saved {pdf_path}", flush=True)
                time.sleep(args.pdf_sleep_seconds)
            else:
                record["download_error"] = "PDF URL was unavailable, blocked, or did not return a PDF"
                print(f"PDF {index}/{total_requested}: failed for {document_id}", flush=True)
                failures.append({"document_id": document_id, "error": record["download_error"], "pdf_url": pdf_url})
        else:
            record["download_error"] = "No PDF URL found"
            print(f"PDF {index}/{total_requested}: no PDF URL for {document_id}", flush=True)
            failures.append({"document_id": document_id, "error": record["download_error"]})

        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata_written += 1

    manifest_path = write_manifest(
        paths,
        step="download_openalex",
        parameters={
            "search": args.search,
            "max_results": args.max_results,
            "per_page": args.per_page,
            "sleep_seconds": args.sleep_seconds,
            "pdf_sleep_seconds": args.pdf_sleep_seconds,
        },
        metrics={
            "metadata_written": metadata_written,
            "pdfs_downloaded": downloaded,
            "pdfs_skipped_existing": skipped,
            "failures": len(failures),
        },
        failures=failures,
    )
    print(f"Wrote {metadata_written} metadata records")
    print(f"Downloaded {downloaded} PDFs; skipped {skipped} existing PDFs")
    if failures:
        print(f"Failures: {len(failures)}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
