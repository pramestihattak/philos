import json
from collections.abc import AsyncGenerator

import httpx
import structlog

from src.config import settings
from src.schemas import SourceChunk
from src.services.concurrency import get_semaphore

log = structlog.get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the user's question using ONLY the provided context below.
If the answer is not found in the context, say: "I don't have information about that in my knowledge base."
Do not make up facts. Do not include source labels or citations in your answer. Be concise.

Context:
{context}
"""


def _build_context(chunks: list[SourceChunk]) -> str:
    parts = [chunk.text for chunk in chunks if chunk.text]
    return "\n\n---\n\n".join(parts)


class LLMService:
    def __init__(self) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.llm_model

    async def generate(
        self,
        query: str,
        context_chunks: list[SourceChunk],
        temperature: float | None = None,
    ) -> tuple[str, str, int]:
        """
        Generate an answer grounded in context_chunks.

        Returns (answer, model_name, tokens_used).
        """
        temp = temperature if temperature is not None else settings.temperature
        context = _build_context(context_chunks)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

        payload = {
            "model": self._model,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "stream": False,
            "options": {"temperature": temp},
        }

        async with get_semaphore():
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()

        message = data.get("message", {})
        raw = message.get("content", "").strip()
        # Collapse newlines into spaces for clean plain-text output
        answer = " ".join(raw.split())
        model = data.get("model", self._model)
        usage = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        log.info("llm generation complete", model=model, tokens=usage)
        return answer, model, usage

    async def generate_stream(
        self,
        query: str,
        context_chunks: list[SourceChunk],
        temperature: float | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream an answer token by token.

        Yields dicts:
          {"type": "token", "content": "..."}  — one per streamed chunk
          {"type": "done", "model": "...", "tokens": N}  — on completion
        """
        temp = temperature if temperature is not None else settings.temperature
        context = _build_context(context_chunks)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

        payload = {
            "model": self._model,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "stream": True,
            "options": {"temperature": temp},
        }

        async with get_semaphore():
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    tokens = 0
                    model = self._model
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield {"type": "token", "content": content}
                        if data.get("done"):
                            model = data.get("model", self._model)
                            tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
                            log.info("llm stream complete", model=model, tokens=tokens)
                            yield {"type": "done", "model": model, "tokens": tokens}
                            return

    async def check_model(self) -> bool:
        """Return True if the LLM model is available in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                names = [m["name"] for m in resp.json().get("models", [])]
                return any(self._model in n for n in names)
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []


llm = LLMService()
