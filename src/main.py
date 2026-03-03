import logging
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI

from src.config import settings
from src.services.concurrency import init_semaphore
from src.services.retriever import retriever

# Configure structured logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    ),
)

log = structlog.get_logger(__name__)


def _start_watcher() -> None:
    """Start the watchdog observer for auto-ingestion if configured."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    import asyncio

    documents_dir = Path(settings.documents_dir)

    class IngestHandler(FileSystemEventHandler):
        def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            filepath = Path(str(event.src_path))
            from src.services.ingester import SUPPORTED_EXTENSIONS
            if filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
                log.info("watcher: new file detected", path=str(filepath))
                from src.services.ingester import ingest_file
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(ingest_file(filepath))
                    log.info("watcher: ingested", filename=result.filename, status=result.status)
                finally:
                    loop.close()

    observer = Observer()
    observer.schedule(IngestHandler(), str(documents_dir), recursive=False)
    observer.start()
    log.info("document watcher started", directory=str(documents_dir))
    return observer  # type: ignore[return-value]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    log.info("philos starting up", llm_model=settings.llm_model, embed_model=settings.embed_model)

    # Ensure data dirs exist
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.documents_dir).mkdir(parents=True, exist_ok=True)

    # Init ChromaDB
    retriever.init()

    # Init Ollama concurrency semaphore
    init_semaphore()

    # Verify Ollama is reachable
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
        log.info("ollama reachable", url=settings.ollama_base_url)
    except Exception as exc:
        log.warning("ollama not reachable at startup — inference will fail until it is", error=str(exc))

    # Optionally start file watcher
    observer = None
    if settings.watch_documents_dir:
        observer = _start_watcher()

    yield  # ── App running ──────────────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    if observer is not None:
        observer.stop()
        observer.join()
    log.info("philos shut down")


app = FastAPI(
    title="Philos",
    description="Personal RAG + LLM service powered by Ollama and ChromaDB",
    version="0.1.0",
    lifespan=lifespan,
)

from src.api.routes import router  # noqa: E402 — after app creation to avoid circular imports
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
