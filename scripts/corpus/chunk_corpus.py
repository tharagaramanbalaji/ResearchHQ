from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from rag_corpus.manifest import write_manifest
from rag_corpus.paths import corpus_paths


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def choose_sentence_boundary(spans: list[tuple[str, int, int]], start: int, max_end: int, min_end: int) -> int:
    for index in range(max_end - 1, min_end - 1, -1):
        token = spans[index][0]
        if token.endswith((".", "?", "!", ":", ";")):
            return index + 1
    return max_end


def chunk_text(
    text: str,
    *,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, int | str]]:
    if min_tokens <= 0 or max_tokens <= 0:
        raise ValueError("Token limits must be positive")
    if min_tokens > max_tokens:
        raise ValueError("--min-tokens must be smaller than or equal to --max-tokens")
    if overlap_tokens < 100:
        raise ValueError("--overlap-tokens must be at least 100")
    if overlap_tokens >= min_tokens:
        raise ValueError("--overlap-tokens must be smaller than --min-tokens")

    normalized = normalize_text(text)
    spans = token_spans(normalized)
    chunks: list[dict[str, int | str]] = []
    start_token = 0

    while start_token < len(spans):
        remaining_tokens = len(spans) - start_token
        if remaining_tokens < min_tokens and chunks:
            adjusted_start = max(0, len(spans) - min_tokens)
            if adjusted_start < start_token:
                start_token = adjusted_start
                remaining_tokens = len(spans) - start_token

        if remaining_tokens <= max_tokens:
            end_token = len(spans)
        else:
            max_end = min(start_token + max_tokens, len(spans))
            min_end = min(start_token + min_tokens, max_end)
            end_token = choose_sentence_boundary(spans, start_token, max_end, min_end)

        char_start = spans[start_token][1]
        char_end = spans[end_token - 1][2]
        chunk = normalized[char_start:char_end].strip()

        if chunk:
            chunks.append(
                {
                    "text": chunk,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_count_estimate": end_token - start_token,
                }
            )

        if end_token >= len(spans):
            break
        start_token = max(0, end_token - overlap_tokens)

    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk extracted corpus documents into JSONL records.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--documents-file", default=None)
    parser.add_argument("--chunks-file", default=None)
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = corpus_paths(args.data_dir)
    documents_file = Path(args.documents_file) if args.documents_file else paths.documents_jsonl
    chunks_file = Path(args.chunks_file) if args.chunks_file else paths.chunks_jsonl
    chunks_file.parent.mkdir(parents=True, exist_ok=True)

    documents_seen = 0
    chunks_written = 0

    with documents_file.open("r", encoding="utf-8") as source, chunks_file.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            document = json.loads(line)
            documents_seen += 1

            base_metadata = {
                "source": document.get("source"),
                "source_type": document.get("source_type"),
                "title": document.get("title"),
                "authors": document.get("authors", []),
                "published_at": document.get("published_at"),
                "updated_at": document.get("updated_at"),
                "external_id": document.get("external_id"),
                "url": document.get("url"),
                "path": document.get("path"),
                "pdf_path": document.get("pdf_path"),
            }

            text_parts = [document.get("title", ""), document.get("abstract", ""), document.get("text", "")]
            combined_text = "\n\n".join(part for part in text_parts if part)

            for index, chunk in enumerate(
                chunk_text(
                    combined_text,
                    min_tokens=args.min_tokens,
                    max_tokens=args.max_tokens,
                    overlap_tokens=args.overlap_tokens,
                )
            ):
                record = {
                    "chunk_id": f"{document['document_id']}::chunk_{index:04d}",
                    "document_id": document["document_id"],
                    "chunk_index": index,
                    "text": chunk["text"],
                    "token_count_estimate": chunk["token_count_estimate"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "metadata": base_metadata,
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunks_written += 1

    manifest_path = write_manifest(
        paths,
        step="chunk_corpus",
        parameters={
            "documents_file": str(documents_file),
            "chunks_file": str(chunks_file),
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap_tokens,
        },
        metrics={"documents_seen": documents_seen, "chunks_written": chunks_written},
    )
    print(f"Wrote {chunks_written} chunks from {documents_seen} documents to {chunks_file}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
