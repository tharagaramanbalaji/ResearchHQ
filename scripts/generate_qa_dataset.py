#!/usr/bin/env python3
"""
generate_qa_dataset.py
======================
Generates an evaluation Q&A dataset from the first 50 PDF papers in:
    data/llm_ai_2023_2026/raw/papers/

Strategy:
- First 50 papers only
- 2-3 questions per paper  →  ~100-150 Q&A pairs total
- Context extraction is MIXED — not always the beginning of the paper.
  For each paper we sample 3 distinct "windows" from different parts of
  the document (intro/abstract, mid-body, results/conclusion), then ask
  Gemini to generate 2-3 specific questions from ALL windows combined.
- Model: gemini-3.1-flash-lite

Outputs:
    data/evaluation/qa_dataset.json   — machine-readable (for eval scripts)
    data/evaluation/qa_dataset.md     — human-readable review copy

Usage (PowerShell):
    $env:GEMINI_API_KEY = "your-key"
    .\\venv\\Scripts\\python.exe scripts\\generate_qa_dataset.py

    # Resume after interruption:
    .\\venv\\Scripts\\python.exe scripts\\generate_qa_dataset.py --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR   = PROJECT_ROOT / "data" / "llm_ai_2023_2026" / "raw" / "papers"
EVAL_DIR     = PROJECT_ROOT / "data" / "evaluation"
OUTPUT_JSON  = EVAL_DIR / "qa_dataset.json"
OUTPUT_MD    = EVAL_DIR / "qa_dataset.md"

# ── model ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3.1-flash-lite"

# How many PDFs to process
MAX_PAPERS = 150

# How many questions to request per paper (Gemini may return 2 or 3)
QA_PER_PAPER_MIN = 2
QA_PER_PAPER_MAX = 3

# Target window size for each context slice (chars)
WINDOW_CHARS = 1800


# ─────────────────────────────────────────────────────────────────────────────
# PDF extraction with MIXED context windows
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_pages(pdf_path: Path) -> tuple[str, str]:
    """
    Extract all text from a PDF.
    Returns (title, full_text).
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("ERROR: pypdf not installed. Run: pip install pypdf", file=sys.stderr)
        sys.exit(1)

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        return pdf_path.stem, ""

    pages: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
            if t.strip():
                pages.append(t.strip())
        except Exception:
            pass

    full_text = "\n\n".join(pages)

    # Detect title — longest line in first 8 non-empty lines of page 0
    title = pdf_path.stem
    if pages:
        lines = [l.strip() for l in pages[0].split("\n") if l.strip()]
        candidates = [l for l in lines[:8] if 20 < len(l) < 200]
        if candidates:
            title = max(candidates, key=len)
        elif lines:
            title = lines[0]

    return title, full_text


def build_mixed_context(full_text: str, window: int = WINDOW_CHARS) -> str:
    """
    Sample 3 non-overlapping windows from different parts of the document:
      - Window A: beginning  (intro / abstract area)
      - Window B: middle     (methods / experiments section)
      - Window C: end        (results / conclusion area)

    Each window is snapped to a sentence boundary. The three windows are
    labelled and concatenated so the model can generate diverse questions
    from different parts of the paper.
    """
    text = full_text.strip()
    total = len(text)
    if total == 0:
        return ""

    def snap_to_sentence(start: int, end: int) -> str:
        """Extract text[start:end] snapped outward to sentence boundaries."""
        # Snap start backward to previous sentence end
        chunk = text[max(0, start):min(total, end)]
        # Try to start after a period+space
        first_period = chunk.find(". ")
        if 0 < first_period < len(chunk) // 3:
            chunk = chunk[first_period + 2:]
        # Snap end to last sentence
        last_period = chunk.rfind(".")
        if last_period > len(chunk) // 2:
            chunk = chunk[: last_period + 1]
        return chunk.strip()

    # Three target centre points spread across the document
    centres = [
        int(total * 0.10),   # near beginning (~abstract / intro)
        int(total * 0.48),   # near middle   (~methods / experiments)
        int(total * 0.82),   # near end      (~results / conclusion)
    ]

    labelled_sections: list[str] = []
    labels = ["[INTRO/ABSTRACT]", "[METHODS/EXPERIMENTS]", "[RESULTS/CONCLUSION]"]

    for label, centre in zip(labels, centres):
        half = window // 2
        raw_start = max(0, centre - half)
        raw_end   = min(total, centre + half)
        chunk = snap_to_sentence(raw_start, raw_end)
        if chunk:
            labelled_sections.append(f"{label}\n{chunk}")

    return "\n\n".join(labelled_sections)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini generation
# ─────────────────────────────────────────────────────────────────────────────

GENERATION_PROMPT = """\
You are an expert research assistant building a retrieval-augmented generation (RAG) evaluation dataset.

Below are THREE excerpts from DIFFERENT sections of an academic paper (intro/abstract, methods, results).
Generate exactly {n_questions} specific, non-trivial questions that can be answered from these excerpts.

Rules for questions:
- Each question must be answerable using ONLY the text provided — no outside knowledge.
- Prefer questions about specific claims, metrics, method names, benchmark numbers, or key findings.
- Avoid vague questions like "What is this paper about?" or "What does the paper conclude?"
- Each question must be DIFFERENT from the others (cover different facts / sections).

Rules for answers:
- Each answer must be 2-4 sentences, grounded ONLY in the excerpts.
- Do NOT hallucinate anything not stated in the text.

Output ONLY a valid JSON array of exactly {n_questions} objects, each with keys "question" and "answer":
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

Paper excerpts:
---
ARXIV ID: {arxiv_id}
{paper_text}
---

JSON array output:"""


def generate_qa_pairs(
    client,
    arxiv_id: str,
    paper_text: str,
    n_questions: int = 3,
    max_retries: int = 4,
) -> list[dict] | None:
    """
    Call Gemini to generate n_questions Q&A pairs from the paper excerpt.
    Returns a list of dicts with 'question' and 'answer' keys, or None on total failure.
    """
    prompt = GENERATION_PROMPT.format(
        n_questions=n_questions,
        arxiv_id=arxiv_id,
        paper_text=paper_text,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            raw = response.text.strip()

            # Strip markdown code fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
            raw = raw.strip()

            parsed = json.loads(raw)

            # Accept either a list or a dict with a list value
            if isinstance(parsed, dict):
                # Sometimes model wraps it: {"questions": [...]}
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break

            if not isinstance(parsed, list):
                print(f"  [WARN] Expected list, got {type(parsed)} for {arxiv_id}")
                continue

            results = []
            for item in parsed:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    q = item["question"].strip()
                    a = item["answer"].strip()
                    if q and a:
                        results.append({"question": q, "answer": a})

            if results:
                return results

            print(f"  [WARN] No valid Q&A items extracted from response for {arxiv_id}")

        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error for {arxiv_id} (attempt {attempt}/{max_retries}): {e}")
            if 'raw' in dir():
                print(f"         Raw (first 400 chars): {raw[:400]}")
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                wait = 30 * attempt
                print(f"  [RATE LIMIT] Waiting {wait}s before retry {attempt}/{max_retries}...")
                time.sleep(wait)
            elif "503" in err_str or "overloaded" in err_str.lower():
                wait = 15 * attempt
                print(f"  [OVERLOADED] Waiting {wait}s before retry {attempt}/{max_retries}...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] Gemini call failed for {arxiv_id} (attempt {attempt}): {e}")
                time.sleep(5)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(qa_entries: list[dict]) -> None:
    """Write the Q&A dataset to JSON and Markdown."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(qa_entries, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON  → {OUTPUT_JSON}  ({len(qa_entries)} entries)")

    # ── Markdown ──────────────────────────────────────────────────────────────
    total = len(qa_entries)
    unique_papers = len({e["arxiv_id"] for e in qa_entries})
    header = [
        "# RAG Evaluation Dataset",
        "",
        f"**Total Q&A pairs:** {total}  ",
        f"**Papers covered:** {unique_papers}  ",
        f"**Model used:** {GEMINI_MODEL}  ",
        f"**Context strategy:** Mixed windows (intro, methods, results)  ",
        "",
        "---",
        "",
    ]

    body: list[str] = []
    for i, entry in enumerate(qa_entries, 1):
        q_num = entry.get("question_number", i)
        body += [
            f"## Q{i} — Paper: {entry.get('title', entry['arxiv_id'])[:90]}",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| **Source** | `{entry['arxiv_id']}` |",
            f"| **Q# in paper** | {q_num} |",
            f"| **Context window** | {entry.get('context_window', 'mixed')} |",
            "",
            f"**Question:**",
            f"> {entry['question']}",
            "",
            f"**Reference Answer:**",
            f"> {entry['answer']}",
            "",
            "---",
            "",
        ]

    with OUTPUT_MD.open("w", encoding="utf-8") as f:
        f.write("\n".join(header + body))
    print(f"  ✓ MD    → {OUTPUT_MD}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Generate ~{MAX_PAPERS * QA_PER_PAPER_MIN}-{MAX_PAPERS * QA_PER_PAPER_MAX} Q&A pairs from first {MAX_PAPERS} arXiv PDFs."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip papers already in the output JSON (resume interrupted run).",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of questions per paper (default: 3).",
    )
    args = parser.parse_args()
    n_questions = args.n_questions

    # ── Gemini client ─────────────────────────────────────────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        print("  PowerShell:  $env:GEMINI_API_KEY = 'your-key'", file=sys.stderr)
        sys.exit(1)

    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
        sys.exit(1)

    # Quick model sanity check
    print(f"Model  : {GEMINI_MODEL}")
    print(f"Papers : first {MAX_PAPERS} PDFs")
    print(f"Q/paper: {n_questions}")

    # ── Discover PDFs (first 50) ──────────────────────────────────────────────
    all_pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not all_pdfs:
        print(f"ERROR: No PDFs found in {PAPERS_DIR}", file=sys.stderr)
        sys.exit(1)

    pdf_files = all_pdfs[:MAX_PAPERS]
    print(f"\nFound {len(all_pdfs)} total PDFs → using first {len(pdf_files)}\n")

    # ── Resume: load existing entries keyed by arxiv_id ──────────────────────
    done_arxiv_ids: set[str] = set()
    existing_entries: list[dict] = []
    if args.resume and OUTPUT_JSON.exists():
        with OUTPUT_JSON.open("r", encoding="utf-8") as f:
            existing_entries = json.load(f)
        done_arxiv_ids = {e["arxiv_id"] for e in existing_entries}
        print(f"Resuming — {len(done_arxiv_ids)} papers already done ({len(existing_entries)} Q&A pairs).\n")

    # ── Process each paper ────────────────────────────────────────────────────
    qa_entries: list[dict] = list(existing_entries)  # start from existing
    failed: list[str] = []
    global_q_id = len(existing_entries) + 1

    for paper_idx, pdf_path in enumerate(pdf_files, 1):
        arxiv_id = pdf_path.stem.replace("arxiv_", "").replace("v1", "")

        if arxiv_id in done_arxiv_ids:
            print(f"[{paper_idx:2}/{len(pdf_files)}] {arxiv_id}  →  skip (already done)")
            continue

        print(f"[{paper_idx:2}/{len(pdf_files)}] {arxiv_id}")

        # 1. Extract full text
        title, full_text = extract_all_pages(pdf_path)
        if not full_text.strip():
            print(f"  [WARN] No text extracted — skipping")
            failed.append(arxiv_id)
            continue

        print(f"  Title : {title[:80]}")
        print(f"  Text  : {len(full_text):,} chars total")

        # 2. Build MIXED context (intro + methods + results windows)
        mixed_ctx = build_mixed_context(full_text, window=WINDOW_CHARS)
        if not mixed_ctx.strip():
            print(f"  [WARN] Context windows empty — skipping")
            failed.append(arxiv_id)
            continue

        print(f"  Ctx   : {len(mixed_ctx):,} chars (3 mixed windows)")

        # 3. Generate 2-3 Q&A pairs from the mixed context
        print(f"  → Calling {GEMINI_MODEL} (requesting {n_questions} questions)...")
        pairs = generate_qa_pairs(client, arxiv_id, mixed_ctx, n_questions=n_questions)

        if not pairs:
            print(f"  [FAIL] Could not generate Q&A for {arxiv_id}")
            failed.append(arxiv_id)
            continue

        # 4. Add entries
        for q_num, pair in enumerate(pairs, 1):
            entry = {
                "id": global_q_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "question_number": q_num,   # 1, 2, or 3 within the paper
                "context_window": "mixed (intro+methods+results)",
                "question": pair["question"],
                "answer": pair["answer"],
            }
            qa_entries.append(entry)
            global_q_id += 1
            print(f"  Q{q_num}: {pair['question'][:100]}...")

        done_arxiv_ids.add(arxiv_id)

        # Polite delay between papers
        time.sleep(1.2)

        # Checkpoint every 10 papers
        if paper_idx % 10 == 0:
            print(f"\n  ── Checkpoint at paper {paper_idx} ({len(qa_entries)} Q&A total) ──")
            write_outputs(qa_entries)

    # ── Final output ──────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"Generation complete!")
    print(f"  Papers processed : {len(done_arxiv_ids)}")
    print(f"  Q&A pairs        : {len(qa_entries)}")
    print(f"  Failed papers    : {len(failed)}" + (f"  → {failed}" if failed else ""))

    write_outputs(qa_entries)

    print(f"\nTo run your evaluation:")
    print(f"  JSON : {OUTPUT_JSON}")
    print(f"  MD   : {OUTPUT_MD}")


if __name__ == "__main__":
    main()
