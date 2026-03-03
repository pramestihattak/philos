import httpx
import structlog

from src.config import settings
from src.services.concurrency import get_semaphore

log = structlog.get_logger(__name__)


class EmbedderService:
    def __init__(self) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.embed_model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a float vector."""
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns a list of float vectors."""
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                async with get_semaphore():
                    resp = await client.post(
                        f"{self._base_url}/api/embeddings",
                        json={"model": self._model, "prompt": text},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                vectors.append(data["embedding"])
        log.debug("embedded texts", count=len(texts), model=self._model)
        return vectors

    async def check_model(self) -> bool:
        """Return True if the embed model is available in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                names = [m["name"] for m in resp.json().get("models", [])]
                return any(self._model in n for n in names)
        except Exception:
            return False


embedder = EmbedderService()
