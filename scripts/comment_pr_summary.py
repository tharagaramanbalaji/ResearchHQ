#!/usr/bin/env python3
# Test change to trigger PR evaluation gate
"""
comment_pr_summary.py
=====================
Reads RAG evaluation summary from data/evaluation/ragas_summary.json and posts
or updates a formatted status comment on the GitHub Pull Request.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_JSON = PROJECT_ROOT / "data" / "evaluation" / "ragas_summary.json"
COMMENT_SIGNATURE = "<!-- RAG_EVAL_SUMMARY_COMMENT -->"


def format_markdown_report(summary: dict) -> str:
    faithfulness = summary.get("faithfulness", 0.0)
    answer_relevancy = summary.get("answer_relevancy", 0.0)
    n_evaluated = summary.get("n_evaluated", 0)
    total_questions = summary.get("total_questions", 0)
    skipped_total = summary.get("skipped_total", 0)
    skipped_declined = summary.get("skipped_declined", 0)
    skipped_no_context = summary.get("skipped_no_context", 0)
    skipped_api_error = summary.get("skipped_api_error", 0)
    
    thresholds = summary.get("thresholds", {})
    min_faithfulness = thresholds.get("min_faithfulness", 0.80)
    min_answer_relevancy = thresholds.get("min_answer_relevancy", 0.75)
    min_evaluated = thresholds.get("min_evaluated", 100)

    # Determine status
    passed = (
        faithfulness >= min_faithfulness
        and answer_relevancy >= min_answer_relevancy
        and n_evaluated >= min_evaluated
    )
    status_icon = "🟢 **PASS**" if passed else "🔴 **FAIL**"
    
    # Tables
    md = [
        COMMENT_SIGNATURE,
        f"## 🤖 RAG Evaluation Gate Results — {status_icon}",
        "",
        "| Metric | Score | Target | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Faithfulness** | {faithfulness:.3f} | >= {min_faithfulness:.3f} | {'✅' if faithfulness >= min_faithfulness else '❌'} |",
        f"| **Answer Relevancy** | {answer_relevancy:.3f} | >= {min_answer_relevancy:.3f} | {'✅' if answer_relevancy >= min_answer_relevancy else '❌'} |",
        f"| **Evaluated Count** | {n_evaluated} | >= {min_evaluated} | {'✅' if n_evaluated >= min_evaluated else '❌'} |",
        "",
        "### 📊 Execution Details",
        "",
        f"- **Total Questions in Dataset**: {total_questions}",
        f"- **Evaluated Successfully**: {n_evaluated}",
        f"- **Total Skipped**: {skipped_total}",
        f"  - *RAG Declined*: {skipped_declined}",
        f"  - *No Context Retrieved*: {skipped_no_context}",
        f"  - *API / Generation Errors*: {skipped_api_error}",
        "",
    ]

    low_faith = summary.get("low_faithfulness", [])
    if low_faith:
        md.append("### ⚠️ Low Faithfulness Samples")
        md.append("The following questions had faithfulness scores < 0.50:")
        for q in low_faith[:5]:
            md.append(f"- *{q}*")
        if len(low_faith) > 5:
            md.append(f"\n*(and {len(low_faith) - 5} more; see full logs in artifacts)*")
        md.append("")

    md.append("---")
    md.append("*Report generated automatically by RAG evaluation pipeline CI.*")
    return "\n".join(md)


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    if not SUMMARY_JSON.exists():
        print(f"Error: summary file not found at {SUMMARY_JSON}")
        sys.exit(1)

    with SUMMARY_JSON.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    report_body = format_markdown_report(summary)

    # If running locally, print report to console and exit
    if not (token and repo and event_path):
        print("\n=== Local Markdown Report ===")
        print(report_body)
        print("=============================")
        print("Skipping GitHub API comment (missing GITHUB_TOKEN, GITHUB_REPOSITORY, or GITHUB_EVENT_PATH)")
        sys.exit(0)

    # Parse Event Path to get PR Number
    try:
        with open(event_path, "r", encoding="utf-8") as f:
            event = json.load(f)
        pr_number = event.get("pull_request", {}).get("number")
    except Exception as e:
        print(f"Could not parse event payload at {event_path}: {e}")
        pr_number = None

    if not pr_number:
        print("Not a pull request context (no PR number found). Skipping comment.")
        sys.exit(0)

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    
    # 1. Fetch existing comments to search for the signature
    comments_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    try:
        resp = requests.get(comments_url, headers=headers, timeout=30)
        resp.raise_for_status()
        comments = resp.json()
    except Exception as e:
        print(f"Error listing PR comments: {e}")
        sys.exit(1)

    existing_comment_id = None
    for comment in comments:
        if COMMENT_SIGNATURE in comment.get("body", ""):
            existing_comment_id = comment["id"]
            break

    # 2. Post or update the comment
    if existing_comment_id:
        update_url = f"https://api.github.com/repos/{repo}/issues/comments/{existing_comment_id}"
        print(f"Updating existing comment {existing_comment_id}...")
        try:
            resp = requests.patch(update_url, headers=headers, json={"body": report_body}, timeout=30)
            resp.raise_for_status()
            print("Successfully updated PR comment.")
        except Exception as e:
            print(f"Error updating comment: {e}")
            sys.exit(1)
    else:
        print("Creating new PR comment...")
        try:
            resp = requests.post(comments_url, headers=headers, json={"body": report_body}, timeout=30)
            resp.raise_for_status()
            print("Successfully posted new PR comment.")
        except Exception as e:
            print(f"Error creating comment: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
