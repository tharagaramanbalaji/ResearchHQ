from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pypdf import PdfReader

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from rag_corpus.files import sha256_file
from rag_corpus.manifest import write_manifest
from rag_corpus.paths import corpus_paths


def extract_text(pdf_path: Path) -> tuple[str, int]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # Clean lone surrogate characters that are invalid in UTF-8
        text = text.encode("utf-8", "ignore").decode("utf-8")
        pages.append(text)
    return "\n\n".join(pages).strip(), len(reader.pages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text from downloaded PDFs into document JSONL.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--documents-file", default=None)
    parser.add_argument("--min-text-chars", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = corpus_paths(args.data_dir)
    documents_file = Path(args.documents_file) if args.documents_file else paths.documents_jsonl
    documents_file.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failures: list[dict[str, str]] = []

    metadata_files = sorted(paths.metadata_dir.glob("*.json"))
    total_files = len(metadata_files)
    print(f"Starting text extraction from {total_files} PDFs...", flush=True)

    with documents_file.open("w", encoding="utf-8") as sink:
        for index, metadata_path in enumerate(metadata_files, 1):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            doc_id = metadata.get("document_id", metadata_path.stem)
            print(f"[{index}/{total_files}] Extracting text from {doc_id}...", flush=True)
            pdf_path = Path(metadata.get("pdf_path", ""))

            if not pdf_path.exists():
                skipped += 1
                failures.append({"document_id": metadata.get("document_id", metadata_path.stem), "error": "missing PDF"})
                continue

            try:
                text, page_count = extract_text(pdf_path)
            except Exception as exc:
                skipped += 1
                failures.append({"document_id": metadata.get("document_id", metadata_path.stem), "error": str(exc)})
                continue

            if len(text) < args.min_text_chars:
                skipped += 1
                failures.append(
                    {
                        "document_id": metadata.get("document_id", metadata_path.stem),
                        "error": f"extracted text below {args.min_text_chars} characters",
                    }
                )
                continue

            record = {
                **metadata,
                "source_type": metadata.get("source_type", "pdf"),
                "pdf_path": str(pdf_path),
                "sha256": metadata.get("sha256") or sha256_file(pdf_path),
                "page_count": page_count,
                "text": text,
            }
            json_str = json.dumps(record, ensure_ascii=False)
            # Ensure no surrogate characters are written to the UTF-8 output file
            json_str = json_str.encode("utf-8", "ignore").decode("utf-8")
            sink.write(json_str + "\n")
            processed += 1

    manifest_path = write_manifest(
        paths,
        step="extract_pdf_text",
        parameters={
            "documents_file": str(documents_file),
            "min_text_chars": args.min_text_chars,
        },
        metrics={"documents_processed": processed, "documents_skipped": skipped, "failures": len(failures)},
        failures=failures,
    )
    print(f"Wrote {processed} documents to {documents_file}")
    print(f"Skipped {skipped} documents")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
