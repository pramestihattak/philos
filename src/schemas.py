from typing import Any
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question or prompt to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of document chunks to retrieve")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="LLM sampling temperature")
    include_sources: bool = Field(False, description="Include source chunk text in response")


class SourceChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    text: str | None = None  # Only populated when include_sources=True
    score: float


class InferenceResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str
    tokens: int


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_created: int
    status: str  # "ingested" | "skipped" | "error"
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    chroma: bool
    models: dict[str, Any] = {}
