# philos

Personal RAG + LLM service. Feed it your documents, ask it questions, get answers grounded in your own knowledge base — fully local, no cloud APIs.

**Stack:** Ollama (qwen3.5:9b + nomic-embed-text) · ChromaDB · FastAPI

---

## Setup

### 1. Install Ollama and pull models

```bash
brew install ollama
ollama serve   # keep running in a separate terminal

ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure (optional)

```bash
cp .env.example .env
# Edit .env to change models, paths, chunk size, etc.
```

### 4. Add documents

Drop `.pdf`, `.txt`, `.md`, or `.docx` files into `data/documents/`.

### 5. Ingest documents

```bash
# Ingest everything in data/documents/
python scripts/ingest.py data/documents/ --recursive

# Ingest a single file
python scripts/ingest.py path/to/notes.pdf

# List what's in the knowledge base
python scripts/ingest.py --list

# Delete a document (use the doc_id from --list)
python scripts/ingest.py --delete <doc_id>

# Force re-ingest (e.g. after updating a file)
python scripts/ingest.py data/documents/ --recursive --force
```

### 6. Start the server

```bash
uvicorn src.main:app --reload --port 8000
```

---

## API

### Authentication

Auth is disabled by default (safe for localhost). To require a Bearer token, set `API_KEY` in `.env`:

```bash
API_KEY=your-secret-key
```

All requests then need:

```
Authorization: Bearer your-secret-key
```

Requests without the header, or with a wrong key, return `401 Unauthorized`.

### `POST /inference`

Ask a question. The service embeds your query, retrieves the most relevant chunks, and generates a grounded answer.

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{"query": "What are the key points from my notes?"}'
```

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Your question |
| `top_k` | int | 5 | Number of chunks to retrieve |
| `temperature` | float | 0.1 | LLM sampling temperature (0=deterministic) |
| `include_sources` | bool | false | Include raw chunk text in response |

**Response:**

```json
{
  "answer": "Based on your notes, the key points are...",
  "sources": [
    {"doc_id": "abc123", "filename": "notes.pdf", "chunk_index": 2, "score": 0.91}
  ],
  "model": "qwen3.5:9b",
  "tokens": 312
}
```

### `POST /inference/stream`

Same as `/inference` but streams tokens as they are generated using [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events). Useful when the full response takes several seconds and you want to show output in real time.

```bash
# -N disables curl's output buffering
curl -N -X POST http://localhost:8000/inference/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{"query": "What are the key points from my notes?"}'
```

**Response** (`Content-Type: text/event-stream`):

```
data: {"type":"token","content":"Based"}
data: {"type":"token","content":" on"}
data: {"type":"token","content":" your notes"}
...
data: {"type":"done","answer":"Based on your notes...","sources":[...],"model":"qwen3.5:9b","tokens":312}
```

On error: `data: {"type":"error","message":"..."}`.

### `POST /documents/ingest`

Upload files directly via the API.

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -H "Authorization: Bearer your-secret-key" \
  -F "files=@path/to/document.pdf" \
  -F "files=@path/to/notes.txt"
```

### `GET /documents`

List all ingested documents.

### `DELETE /documents/{doc_id}`

Remove a document from the knowledge base.

### `GET /health`

Check that Ollama and ChromaDB are reachable and models are available.

```bash
curl http://localhost:8000/health \
  -H "Authorization: Bearer your-secret-key"
```

---

## Configuration

All settings can be overridden via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `qwen3.5:9b` | Generation model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./data/vectorstore` | ChromaDB storage path |
| `DOCUMENTS_DIR` | `./data/documents` | Document drop folder |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Default retrieval count |
| `TEMPERATURE` | `0.1` | Default LLM temperature |
| `API_KEY` | `` | Bearer token for auth; empty = disabled |
| `OLLAMA_CONCURRENCY` | `1` | Max concurrent Ollama requests (embedder + LLM) |
| `WATCH_DOCUMENTS_DIR` | `false` | Auto-ingest new files in `DOCUMENTS_DIR` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ANONYMIZED_TELEMETRY` | `True` | Set to `False` to suppress ChromaDB telemetry warnings |

---

## Architecture

```
User Query → POST /inference
              ↓
         Embed query (nomic-embed-text via Ollama)
              ↓
         Vector search (ChromaDB cosine similarity)
              ↓
         Build context prompt
              ↓
         LLM generation (qwen3.5:9b via Ollama)
              ↓
         Return answer + source citations
```
