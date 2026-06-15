# Enterprise RAG Corpus Pipeline

This project starts with the data creation layer for an ask-my-research-papers system.
The scripts build a repeatable local corpus that can later be loaded into a vector
database, search index, document store, or evaluation pipeline.

## What Gets Created

```text
data/
  raw/
    papers/              # downloaded PDFs
    markdown/            # optional local markdown source area
    web_pages/           # downloaded raw HTML for web sources
    metadata/            # one metadata JSON file per paper
  processed/
    documents.jsonl      # document-level text and metadata
    chunks.jsonl         # retrieval-ready chunks
  manifests/
    runs/                # audit records for each pipeline step
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Recommended Workflow

Download papers from arXiv:

```bash
python scripts/corpus/download_arxiv.py --query "cat:cs.AI OR cat:cs.CL" --max-results 150
```

The arXiv downloader is intentionally single-connection and sequential. It enforces
these API rules:

- At least 3 seconds between arXiv API requests.
- At most 2,000 results per `start` / `max_results` slice.
- At most 30,000 results for one query.

Or use a specific list of arXiv IDs/links:

```bash
python scripts/corpus/download_arxiv.py --arxiv-ids-file ./arxiv_ids.txt --max-results 150
```

`arxiv_ids.txt` can contain IDs or URLs:

```text
2401.00001
https://arxiv.org/abs/2305.18290
https://arxiv.org/pdf/1706.03762.pdf
```

Ingest local PDFs, Markdown files, and web pages:

```bash
python scripts/corpus/ingest_documents.py --pdf-dir ./my_pdfs --markdown-dir ./notes --web-urls-file ./urls.txt
```

Extract arXiv PDF text into document records:

```bash
python scripts/corpus/extract_pdf_text.py
```

Chunk extracted documents for retrieval. The default chunking policy targets
enterprise RAG use: 500-800 estimated tokens per chunk with at least 100 tokens
of overlap between adjacent chunks.

```bash
python scripts/corpus/chunk_corpus.py --min-tokens 500 --max-tokens 800 --overlap-tokens 100
```

Or run the whole pipeline:

```bash
python scripts/corpus/build_corpus.py --query "cat:cs.AI OR cat:cs.CL" --max-results 25
```

For an arXiv-only starter corpus of 100-150 papers, use:

```bash
python scripts/corpus/build_corpus.py --query "cat:cs.AI OR cat:cs.CL" --max-results 150
```

For LLM/AI papers submitted in the last 3 years, use an arXiv category and
keyword query with a submitted-date filter. As of June 3, 2026, the last 3 years
starts on June 3, 2023:

```bash
python scripts/corpus/build_corpus.py ^
  --query "(cat:cs.AI OR cat:cs.CL OR cat:cs.LG) AND (all:LLM OR all:\"large language model\" OR all:\"large language models\" OR all:transformer OR all:generative)" ^
  --from-date 2023-06-03 ^
  --to-date 2026-06-03 ^
  --max-results 150 ^
  --batch-size 10 ^
  --sleep-seconds 10 ^
  --pdf-sleep-seconds 5
```

For a handpicked arXiv-only corpus, use:

```bash
python scripts/corpus/build_corpus.py --arxiv-ids-file ./arxiv_ids.txt --max-results 150
```

You can also mix arXiv, local files, and URLs in one run:

```bash
python scripts/corpus/build_corpus.py ^
  --query "cat:cs.AI OR cat:cs.CL" ^
  --max-results 25 ^
  --pdf-dir ./my_pdfs ^
  --markdown-dir ./notes ^
  --web-urls-file ./urls.txt
```

`urls.txt` should contain one URL per line. Blank lines and lines starting with
`#` are ignored.

## Output Schemas

`processed/documents.jsonl` contains one JSON object per source document:

```json
{
  "document_id": "arxiv_2401.00001",
  "source": "arxiv",
  "source_type": "pdf",
  "title": "Paper title",
  "authors": ["Author One"],
  "abstract": "Paper abstract",
  "published_at": "2024-01-01T00:00:00Z",
  "pdf_path": "data/raw/papers/arxiv_2401.00001.pdf",
  "sha256": "...",
  "page_count": 12,
  "text": "Full extracted text"
}
```

`processed/chunks.jsonl` contains retrieval-ready chunks:

```json
{
  "chunk_id": "arxiv_2401.00001::chunk_0000",
  "document_id": "arxiv_2401.00001",
  "chunk_index": 0,
  "text": "Chunk text",
  "token_count_estimate": 318,
  "char_start": 0,
  "char_end": 1398,
  "metadata": {
    "source": "arxiv",
    "source_type": "pdf",
    "title": "Paper title",
    "authors": ["Author One"],
    "published_at": "2024-01-01T00:00:00Z"
  }
}
```

These JSONL files are intentionally database-agnostic. The next layer can transform
`chunks.jsonl` into embeddings and insert records into Pinecone, Weaviate, Qdrant,
pgvector, Elasticsearch/OpenSearch, or another storage backend.
