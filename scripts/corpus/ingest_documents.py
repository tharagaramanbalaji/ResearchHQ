from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from pypdf import PdfReader

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from rag_corpus.files import sha256_file
from rag_corpus.manifest import write_manifest
from rag_corpus.paths import corpus_paths


SUPPORTED_MARKDOWN = {".md", ".markdown", ".mdx"}
SUPPORTED_PDFS = {".pdf"}
WHITESPACE_RE = re.compile(r"\s+")


class ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = normalize_space(html.unescape(data))
        if not cleaned:
            return
        if self._in_title:
            self.title = normalize_space(f"{self.title} {cleaned}")
        self._parts.append(cleaned)

    @property
    def text(self) -> str:
        return normalize_space("\n".join(self._parts))


def normalize_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def document_id_for(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip(), len(reader.pages)


def ingest_pdf(pdf_path: Path) -> dict:
    text, page_count = extract_pdf_text(pdf_path)
    title = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
    return {
        "document_id": document_id_for("pdf", str(pdf_path.resolve())),
        "source": "local_file",
        "source_type": "pdf",
        "title": title,
        "authors": [],
        "abstract": "",
        "path": str(pdf_path),
        "pdf_path": str(pdf_path),
        "sha256": sha256_file(pdf_path),
        "page_count": page_count,
        "text": text,
    }


def ingest_markdown(markdown_path: Path) -> dict:
    text = markdown_path.read_text(encoding="utf-8")
    title = first_markdown_heading(text) or markdown_path.stem.replace("_", " ").replace("-", " ").strip()
    return {
        "document_id": document_id_for("md", str(markdown_path.resolve())),
        "source": "local_file",
        "source_type": "markdown",
        "title": title,
        "authors": [],
        "abstract": "",
        "path": str(markdown_path),
        "sha256": sha256_file(markdown_path),
        "text": text.strip(),
    }


def first_markdown_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None


def fetch_web_page(url: str, timeout: int = 60) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "enterprise-rag-corpus/0.1 (research corpus builder)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read().decode(charset, errors="replace")
    return body, content_type


def ingest_web_url(url: str, raw_output_dir: Path) -> dict:
    html_body, content_type = fetch_web_page(url)
    parser = ReadableHTMLParser()
    parser.feed(html_body)

    document_id = document_id_for("web", url)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_output_dir / f"{document_id}.html"
    raw_path.write_text(html_body, encoding="utf-8")

    parsed = urlparse(url)
    title = parser.title or parsed.netloc + parsed.path
    return {
        "document_id": document_id,
        "source": "web",
        "source_type": "web_page",
        "title": title,
        "authors": [],
        "abstract": "",
        "url": url,
        "raw_html_path": str(raw_path),
        "content_type": content_type,
        "sha256": sha256_file(raw_path),
        "text": parser.text,
    }


def iter_files(root: Path, extensions: set[str]) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in extensions else []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs, Markdown files, and web pages into documents.jsonl.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pdf-dir", action="append", default=[], help="PDF file or directory. Can be passed multiple times.")
    parser.add_argument(
        "--markdown-dir",
        action="append",
        default=[],
        help="Markdown file or directory. Can be passed multiple times.",
    )
    parser.add_argument("--web-urls-file", default=None, help="Text file containing one URL per line.")
    parser.add_argument("--documents-file", default=None)
    parser.add_argument("--append", action="store_true", help="Append to documents.jsonl instead of overwriting it.")
    parser.add_argument("--min-text-chars", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = corpus_paths(args.data_dir)
    paths.ensure()
    documents_file = Path(args.documents_file) if args.documents_file else paths.documents_jsonl
    documents_file.parent.mkdir(parents=True, exist_ok=True)

    documents: list[dict] = []
    failures: list[dict[str, str]] = []

    for value in args.pdf_dir:
        for pdf_path in iter_files(Path(value), SUPPORTED_PDFS):
            try:
                documents.append(ingest_pdf(pdf_path))
            except Exception as exc:
                failures.append({"source": str(pdf_path), "error": str(exc)})

    for value in args.markdown_dir:
        for markdown_path in iter_files(Path(value), SUPPORTED_MARKDOWN):
            try:
                documents.append(ingest_markdown(markdown_path))
            except Exception as exc:
                failures.append({"source": str(markdown_path), "error": str(exc)})

    if args.web_urls_file:
        urls = [
            line.strip()
            for line in Path(args.web_urls_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for url in urls:
            try:
                documents.append(ingest_web_url(url, paths.web_pages_dir))
            except Exception as exc:
                failures.append({"source": url, "error": str(exc)})

    written = 0
    skipped = 0
    mode = "a" if args.append else "w"
    with documents_file.open(mode, encoding="utf-8") as sink:
        for document in documents:
            if len(document.get("text", "")) < args.min_text_chars:
                skipped += 1
                failures.append(
                    {
                        "source": document.get("path") or document.get("url") or document["document_id"],
                        "error": f"extracted text below {args.min_text_chars} characters",
                    }
                )
                continue
            sink.write(json.dumps(document, ensure_ascii=False) + "\n")
            written += 1

    manifest_path = write_manifest(
        paths,
        step="ingest_documents",
        parameters={
            "pdf_dir": args.pdf_dir,
            "markdown_dir": args.markdown_dir,
            "web_urls_file": args.web_urls_file,
            "documents_file": str(documents_file),
            "append": args.append,
            "min_text_chars": args.min_text_chars,
        },
        metrics={"documents_written": written, "documents_skipped": skipped, "failures": len(failures)},
        failures=failures,
    )

    print(f"Wrote {written} documents to {documents_file}")
    print(f"Skipped {skipped} documents")
    if failures:
        print(f"Failures: {len(failures)}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

