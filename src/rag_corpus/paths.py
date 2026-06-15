from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusPaths:
    data_dir: Path
    raw_dir: Path
    papers_dir: Path
    markdown_dir: Path
    web_pages_dir: Path
    metadata_dir: Path
    processed_dir: Path
    manifests_dir: Path
    manifest_runs_dir: Path
    documents_jsonl: Path
    chunks_jsonl: Path

    def ensure(self) -> None:
        for directory in [
            self.raw_dir,
            self.papers_dir,
            self.markdown_dir,
            self.web_pages_dir,
            self.metadata_dir,
            self.processed_dir,
            self.manifests_dir,
            self.manifest_runs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def corpus_paths(data_dir: str | Path) -> CorpusPaths:
    root = Path(data_dir)
    raw = root / "raw"
    processed = root / "processed"
    manifests = root / "manifests"
    return CorpusPaths(
        data_dir=root,
        raw_dir=raw,
        papers_dir=raw / "papers",
        markdown_dir=raw / "markdown",
        web_pages_dir=raw / "web_pages",
        metadata_dir=raw / "metadata",
        processed_dir=processed,
        manifests_dir=manifests,
        manifest_runs_dir=manifests / "runs",
        documents_jsonl=processed / "documents.jsonl",
        chunks_jsonl=processed / "chunks.jsonl",
    )
