# Philos — Claude Code Context

## What this project is
Local personal RAG + LLM service. No cloud APIs. User drops documents into `data/documents/`,
ingests them, then queries via a single `/inference` HTTP endpoint. All inference runs through
Ollama on the local machine (MacBook Pro M3 Pro).

**Stack:** FastAPI · Ollama (qwen3.5:9b + nomic-embed-text) · ChromaDB · Python 3.11+

---

## Running the project

```bash
# Prerequisites (Ollama must be running)
ollama serve
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

# Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ingest documents, start server
python scripts/ingest.py data/documents/ --recursive
uvicorn src.main:app --reload --port 8000
```

---

## Key files

| File | Purpose |
|---|---|
| `src/config.py` | All settings via Pydantic `Settings`, reads `.env` |
| `src/schemas.py` | Pydantic request/response models |
| `src/main.py` | FastAPI app + lifespan (ChromaDB init, Ollama check, optional watcher) |
| `src/services/embedder.py` | Calls Ollama `/api/embeddings` |
| `src/services/retriever.py` | ChromaDB wrapper: add, search, delete, list |
| `src/services/ingester.py` | File → chunk → embed → store pipeline |
| `src/services/llm.py` | Calls Ollama `/api/chat`, injects RAG context into system prompt |
| `src/api/routes.py` | All HTTP routes |
| `scripts/ingest.py` | CLI tool (single Typer command with flags) |

---

## CLI usage

```bash
python scripts/ingest.py data/documents/ --recursive   # ingest directory
python scripts/ingest.py path/to/file.pdf              # ingest single file
python scripts/ingest.py --list                        # list knowledge base
python scripts/ingest.py --delete <doc_id>             # remove a document
python scripts/ingest.py data/documents/ --force       # force re-ingest
```

**Important:** `--list` and `--delete` are flags, not subcommands. `python scripts/ingest.py list`
will fail with "Path not found: list" — always use `--list`.

---

## API endpoints

All endpoints require `Authorization: Bearer <API_KEY>` when `API_KEY` is set in `.env`.
When `API_KEY` is empty (default), all requests pass through — safe for localhost use.

| Method | Path | Description |
|---|---|---|
| `POST` | `/inference` | Main RAG query endpoint (buffered) |
| `POST` | `/inference/stream` | Streaming RAG endpoint (`text/event-stream`) |
| `POST` | `/documents/ingest` | Upload files via multipart form |
| `GET` | `/documents` | List ingested documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document |
| `GET` | `/health` | Ollama + ChromaDB status |

---

## Architecture — request flow

```
POST /inference
  → embed query (nomic-embed-text via Ollama)
  → cosine similarity search (ChromaDB)
  → build context string from top-k chunks
  → inject into system prompt → Ollama /api/chat
  → return { answer, sources, model, tokens }
```

---

## Design decisions & gotchas

- **Singletons** — `embedder`, `retriever`, `llm` are module-level singletons in their respective
  files. `retriever.init()` must be called before use (done in FastAPI lifespan and CLI).
- **doc_id** — `sha256(filename:mtime)[:16]` for disk files; `sha256(filename:md5(content))[:16]`
  for uploads. Ingestion is idempotent by default; use `--force` to re-ingest.
- **System prompt** — instructs the model to answer from context only, with no inline citations.
  Source labels are intentionally stripped from the context fed to the LLM; citation-style
  `[Source N: file.pdf]` outputs from the model are suppressed via prompt instruction.
- **Answer cleaning** — `" ".join(raw.split())` collapses all newlines/whitespace into a single
  space before returning the answer. This is intentional for clean plain-text output.
- **ChromaDB telemetry warnings** — `capture() takes 1 positional argument but 3 were given` is
  a harmless ChromaDB internal bug. Suppress with `ANONYMIZED_TELEMETRY=False` in `.env`.
- **Supported file types** — `.pdf` (pypdf), `.txt`, `.md` (plain text), `.docx` (python-docx).
- **Directory watcher** — disabled by default; enable with `WATCH_DOCUMENTS_DIR=true` in `.env`.
- **Auth** — opt-in Bearer token via `API_KEY` in `.env`. Empty (default) = no auth. Set a value
  to require `Authorization: Bearer <key>` on all requests. Returns 401 on mismatch.
- **Concurrency guard** — `asyncio.Semaphore(OLLAMA_CONCURRENCY)` shared by embedder and LLM.
  Default is 1 (serial). Increase carefully — Ollama on M3 Pro degrades under parallel load.
- **Streaming** — `POST /inference/stream` returns `text/event-stream`. Each SSE line is
  `data: <json>`. Token events: `{"type":"token","content":"..."}`. Final event:
  `{"type":"done","answer":"...","sources":[...],"model":"...","tokens":N}`.
  Semaphore is held for the full stream duration (the HTTP connection is active).

---

## Configuration (`.env`)

Copy `.env.example` to `.env` to override any default:

```
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3.5:9b
EMBED_MODEL=nomic-embed-text
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
TOP_K=5
TEMPERATURE=0.1
ANONYMIZED_TELEMETRY=False
API_KEY=                      # set a value to enable auth; empty = disabled
OLLAMA_CONCURRENCY=1          # max concurrent Ollama requests
```
