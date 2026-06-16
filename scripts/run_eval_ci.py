#!/usr/bin/env python3
"""
run_eval_ci.py
==============
Starts the local RAG API, waits for health, then runs the evaluation script.
This keeps CI workflow YAML small and pushes orchestration logic into versioned code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# Force UTF-8 encoding for standard streams to prevent UnicodeEncodeError on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEALTH_URL = os.environ.get("RAG_EVAL_HEALTH_URL", "http://127.0.0.1:8000/api/health")
DEFAULT_API_URL = os.environ.get("RAG_EVAL_API_URL", "http://127.0.0.1:8000/api/query")


def wait_for_health(health_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "ok":
                return
        except Exception:
            time.sleep(2)
            continue
        time.sleep(2)
    raise RuntimeError(f"API did not become healthy within {timeout_seconds}s: {health_url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL, help="Health endpoint to poll.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Query endpoint for the evaluator.")
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=int(os.environ.get("RAG_EVAL_STARTUP_TIMEOUT", "120")),
        help="Seconds to wait for the API server to become healthy.",
    )
    # Threshold gates
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=float(os.environ.get("RAG_EVAL_MIN_FAITHFULNESS", "0.80")),
        help="Fail if mean faithfulness drops below this threshold.",
    )
    parser.add_argument(
        "--min-answer-relevancy",
        type=float,
        default=float(os.environ.get("RAG_EVAL_MIN_ANSWER_RELEVANCY", "0.75")),
        help="Fail if mean answer relevancy drops below this threshold.",
    )
    parser.add_argument(
        "--min-evaluated",
        type=int,
        default=int(os.environ.get("RAG_EVAL_MIN_EVALUATED", "100")),
        help="Fail if fewer than this many questions are successfully evaluated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit evaluation to the first N questions (0 = all).",
    )
    args, eval_args = parser.parse_known_args()

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env["RAG_EVAL_API_URL"] = args.api_url

    # Start local API server
    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.rag_corpus.api_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    
    # Construct command for evaluate_faithfulness.py
    eval_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_faithfulness.py"),
        "--api-url", args.api_url,
        "--min-faithfulness", str(args.min_faithfulness),
        "--min-answer-relevancy", str(args.min_answer_relevancy),
        "--min-evaluated", str(args.min_evaluated),
        "--limit", str(args.limit),
        *eval_args,
    ]

    print(f"Starting API server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(server_cmd, cwd=PROJECT_ROOT, env=env)

    try:
        print(f"Waiting for API server at {args.health_url}...")
        wait_for_health(args.health_url, args.startup_timeout)
        print("API server is healthy! Running evaluation...")
        
        completed = subprocess.run(eval_cmd, cwd=PROJECT_ROOT, env=env, check=False)
        
        # Parse output summary JSON to perform gate checks in orchestrator
        summary_file = PROJECT_ROOT / "data" / "evaluation" / "ragas_summary.json"
        
        if completed.returncode != 0:
            print(f"\n[ERROR] Evaluation script exited with error code {completed.returncode}.", file=sys.stderr)
            raise SystemExit(completed.returncode)

        if not summary_file.exists():
            print(f"\n[ERROR] Summary file not found at {summary_file}", file=sys.stderr)
            raise SystemExit(1)

        with summary_file.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        faithfulness = summary.get("faithfulness", 0.0)
        answer_relevancy = summary.get("answer_relevancy", 0.0)
        n_evaluated = summary.get("n_evaluated", 0)

        gate_failures = []
        if faithfulness < args.min_faithfulness:
            gate_failures.append(
                f"faithfulness {faithfulness:.3f} < threshold {args.min_faithfulness:.3f}"
            )
        if answer_relevancy < args.min_answer_relevancy:
            gate_failures.append(
                f"answer_relevancy {answer_relevancy:.3f} < threshold {args.min_answer_relevancy:.3f}"
            )
        if n_evaluated < args.min_evaluated:
            gate_failures.append(
                f"n_evaluated {n_evaluated} < threshold {args.min_evaluated}"
            )

        print("\n" + "=" * 60)
        print("GATE CHECK RESULTS SUMMARY")
        print("=" * 60)
        print(f"  Faithfulness     : {faithfulness:.3f}  (target >= {args.min_faithfulness:.3f})")
        print(f"  Answer Relevancy : {answer_relevancy:.3f}  (target >= {args.min_answer_relevancy:.3f})")
        print(f"  Evaluated Count  : {n_evaluated}  (target >= {args.min_evaluated})")
        print(f"  Total Questions  : {summary.get('total_questions', 0)}")
        print(f"  Skipped Count    : {summary.get('skipped_total', 0)}")

        if gate_failures:
            print("\n  GATE STATUS: FAIL")
            for failure in gate_failures:
                print(f"    - {failure}")
            raise SystemExit(1)

        print("\n  GATE STATUS: PASS")
        raise SystemExit(0)

    finally:
        print("Stopping API server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print("API server did not terminate. Killing...")
            server_proc.kill()
            server_proc.wait(timeout=15)


if __name__ == "__main__":
    main()
