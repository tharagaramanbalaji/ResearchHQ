from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rag_corpus.paths import CorpusPaths


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_manifest(
    paths: CorpusPaths,
    *,
    step: str,
    parameters: dict,
    metrics: dict,
    failures: list[dict] | None = None,
) -> Path:
    paths.manifest_runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{step}_{uuid4().hex[:8]}"
    manifest_path = paths.manifest_runs_dir / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "step": step,
        "created_at": utc_now(),
        "parameters": parameters,
        "metrics": metrics,
        "failures": failures or [],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path

