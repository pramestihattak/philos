import json

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import httpx
import structlog

from src.api.deps import verify_api_key
from src.config import settings
from src.schemas import (
    InferenceRequest,
    InferenceResponse,
    IngestResponse,
    HealthResponse,
    SourceChunk,
)
from src.services.embedder import embedder
from src.services.llm import llm
from src.services.retriever import retriever
from src.services.ingester import ingest_bytes

log = structlog.get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/inference", response_model=InferenceResponse)
async def inference(req: InferenceRequest) -> InferenceResponse:
    """Main RAG endpoint: embed query → retrieve → generate → return answer."""
    log.info("inference request", query=req.query[:80])

    # 1. Embed the query
    try:
        query_embedding = await embedder.embed(req.query)
    except Exception as exc:
        log.error("embedding failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Embedding service error: {exc}")

    # 2. Retrieve top-k chunks
    try:
        chunks = retriever.search(query_embedding, top_k=req.top_k)
    except Exception as exc:
        log.error("retrieval failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Retrieval error: {exc}")

    if not chunks:
        return InferenceResponse(
            answer="I don't have any documents in my knowledge base yet. Please ingest some documents first.",
            sources=[],
            model=settings.llm_model,
            tokens=0,
        )

    # 3. Generate answer
    try:
        answer, model, tokens = await llm.generate(
            query=req.query,
            context_chunks=chunks,
            temperature=req.temperature,
        )
    except Exception as exc:
        log.error("llm generation failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"LLM service error: {exc}")

    # 4. Build response — strip chunk text unless include_sources requested
    sources = [
        SourceChunk(
            doc_id=c.doc_id,
            filename=c.filename,
            chunk_index=c.chunk_index,
            text=c.text if req.include_sources else None,
            score=c.score,
        )
        for c in chunks
    ]

    return InferenceResponse(answer=answer, sources=sources, model=model, tokens=tokens)


@router.post("/inference/stream")
async def inference_stream(req: InferenceRequest) -> StreamingResponse:
    """Streaming RAG endpoint: embed → retrieve → stream LLM tokens as SSE."""
    log.info("inference/stream request", query=req.query[:80])

    async def event_generator():
        # 1. Embed query
        try:
            query_embedding = await embedder.embed(req.query)
        except Exception as exc:
            log.error("embedding failed", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'message': f'Embedding error: {exc}'})}\n\n"
            return

        # 2. Retrieve top-k chunks
        try:
            chunks = retriever.search(query_embedding, top_k=req.top_k)
        except Exception as exc:
            log.error("retrieval failed", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'message': f'Retrieval error: {exc}'})}\n\n"
            return

        if not chunks:
            answer = "I don't have any documents in my knowledge base yet. Please ingest some documents first."
            yield f"data: {json.dumps({'type': 'token', 'content': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'answer': answer, 'sources': [], 'model': settings.llm_model, 'tokens': 0})}\n\n"
            return

        # 3. Stream LLM response
        full_answer: list[str] = []
        try:
            async for event in llm.generate_stream(
                query=req.query,
                context_chunks=chunks,
                temperature=req.temperature,
            ):
                if event["type"] == "token":
                    full_answer.append(event["content"])
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "done":
                    sources = [
                        {
                            "doc_id": c.doc_id,
                            "filename": c.filename,
                            "chunk_index": c.chunk_index,
                            "text": c.text if req.include_sources else None,
                            "score": c.score,
                        }
                        for c in chunks
                    ]
                    done_event = {
                        "type": "done",
                        "answer": "".join(full_answer),
                        "sources": sources,
                        "model": event["model"],
                        "tokens": event["tokens"],
                    }
                    yield f"data: {json.dumps(done_event)}\n\n"
        except Exception as exc:
            log.error("llm stream failed", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'message': f'LLM error: {exc}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/documents/ingest", response_model=list[IngestResponse])
async def ingest_documents(files: list[UploadFile] = File(...)) -> list[IngestResponse]:
    """Upload and ingest one or more documents into the knowledge base."""
    results: list[IngestResponse] = []
    for upload in files:
        content = await upload.read()
        filename = upload.filename or "unknown"
        log.info("received upload", filename=filename, size=len(content))
        result = await ingest_bytes(filename, content)
        results.append(result)
    return results


@router.get("/documents", response_model=list[dict])
async def list_documents() -> list[dict]:
    """List all ingested documents in the knowledge base."""
    return retriever.list_documents()


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """Remove all chunks for a document from the knowledge base."""
    deleted = retriever.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document found with id: {doc_id}")
    return {"doc_id": doc_id, "chunks_deleted": deleted}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check service health: Ollama connectivity and ChromaDB status."""
    ollama_ok = False
    models: dict = {}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            ollama_ok = True
            models = {
                "available": available,
                "llm_ready": any(settings.llm_model in n for n in available),
                "embed_ready": any(settings.embed_model in n for n in available),
            }
    except Exception as exc:
        log.warning("ollama health check failed", error=str(exc))

    chroma_ok = retriever.is_healthy()

    overall = "ok" if (ollama_ok and chroma_ok) else "degraded"
    return HealthResponse(status=overall, ollama=ollama_ok, chroma=chroma_ok, models=models)
