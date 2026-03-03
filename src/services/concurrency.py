import asyncio

from src.config import settings

_semaphore: asyncio.Semaphore | None = None


def init_semaphore() -> None:
    """Create the shared Ollama semaphore. Call once inside the async lifespan."""
    global _semaphore
    _semaphore = asyncio.Semaphore(settings.ollama_concurrency)


def get_semaphore() -> asyncio.Semaphore:
    if _semaphore is None:
        raise RuntimeError("Semaphore not initialized — call init_semaphore() first")
    return _semaphore
