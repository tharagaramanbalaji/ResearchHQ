$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtualenv Python not found at $Python. Create the venv and install requirements first."
}

$Arguments = @(
    "scripts/corpus/build_corpus.py",
    "--query",
    '((cat:cs.AI OR cat:cs.CL OR cat:cs.LG) AND (all:LLM OR (all:large AND all:language AND all:model) OR (all:large AND all:language AND all:models) OR all:transformer OR all:generative))',
    "--from-date",
    "2023-06-03",
    "--to-date",
    "2026-06-03",
    "--max-results",
    "50",
    "--sleep-seconds",
    "10",
    "--pdf-sleep-seconds",
    "5",
    "--data-dir",
    "data/llm_ai_2023_2026"
)

Push-Location $ProjectRoot
try {
    & $Python @Arguments
}
finally {
    Pop-Location
}
