import hashlib
import os
from pathlib import Path

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.schemas import IngestResponse
from src.services.embedder import embedder
from src.services.retriever import retriever

log = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _doc_id(filepath: Path) -> str:
    """Stable ID: hash of filename + file mtime (or content hash for uploads)."""
    stat = filepath.stat()
    key = f"{filepath.name}:{stat.st_mtime}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _doc_id_from_content(filename: str, content: bytes) -> str:
    key = f"{filename}:{hashlib.md5(content).hexdigest()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _load_text(filepath: Path) -> str:
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(filepath))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".txt", ".md"):
        return filepath.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".docx":
        from docx import Document
        doc = Document(str(filepath))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _split(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_text(text)


async def ingest_file(filepath: Path, force: bool = False) -> IngestResponse:
    """Ingest a single file into ChromaDB. Idempotent unless force=True."""
    if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return IngestResponse(
            doc_id="",
            filename=filepath.name,
            chunks_created=0,
            status="error",
            message=f"Unsupported file type: {filepath.suffix}",
        )

    doc_id = _doc_id(filepath)

    if not force and retriever.document_exists(doc_id):
        log.info("skipping already-ingested document", filename=filepath.name, doc_id=doc_id)
        return IngestResponse(
            doc_id=doc_id,
            filename=filepath.name,
            chunks_created=0,
            status="skipped",
            message="Document already ingested. Use force=True to re-ingest.",
        )

    try:
        raw_text = _load_text(filepath)
    except Exception as exc:
        log.error("failed to load file", filename=filepath.name, error=str(exc))
        return IngestResponse(
            doc_id=doc_id,
            filename=filepath.name,
            chunks_created=0,
            status="error",
            message=str(exc),
        )

    chunks = _split(raw_text)
    if not chunks:
        return IngestResponse(
            doc_id=doc_id,
            filename=filepath.name,
            chunks_created=0,
            status="error",
            message="No text extracted from document.",
        )

    embeddings = await embedder.embed_batch(chunks)

    chunk_ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "filename": filepath.name, "chunk_index": i}
        for i in range(len(chunks))
    ]

    # If force re-ingest, delete old chunks first
    if force:
        retriever.delete_document(doc_id)

    retriever.add(chunk_ids, embeddings, chunks, metadatas)

    log.info("ingested document", filename=filepath.name, doc_id=doc_id, chunks=len(chunks))
    return IngestResponse(
        doc_id=doc_id,
        filename=filepath.name,
        chunks_created=len(chunks),
        status="ingested",
    )


async def ingest_bytes(filename: str, content: bytes, force: bool = False) -> IngestResponse:
    """Ingest a file from in-memory bytes (for upload endpoint)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return IngestResponse(
            doc_id="",
            filename=filename,
            chunks_created=0,
            status="error",
            message=f"Unsupported file type: {suffix}",
        )

    doc_id = _doc_id_from_content(filename, content)

    if not force and retriever.document_exists(doc_id):
        return IngestResponse(
            doc_id=doc_id,
            filename=filename,
            chunks_created=0,
            status="skipped",
            message="Document already ingested.",
        )

    # Write to temp file for loading
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        raw_text = _load_text(tmp_path)
    except Exception as exc:
        return IngestResponse(
            doc_id=doc_id,
            filename=filename,
            chunks_created=0,
            status="error",
            message=str(exc),
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    chunks = _split(raw_text)
    if not chunks:
        return IngestResponse(
            doc_id=doc_id,
            filename=filename,
            chunks_created=0,
            status="error",
            message="No text extracted from document.",
        )

    embeddings = await embedder.embed_batch(chunks)

    chunk_ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "filename": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    if force:
        retriever.delete_document(doc_id)

    retriever.add(chunk_ids, embeddings, chunks, metadatas)

    log.info("ingested uploaded document", filename=filename, doc_id=doc_id, chunks=len(chunks))
    return IngestResponse(
        doc_id=doc_id,
        filename=filename,
        chunks_created=len(chunks),
        status="ingested",
    )


async def ingest_directory(dirpath: Path, recursive: bool = False) -> list[IngestResponse]:
    """Ingest all supported files in a directory."""
    pattern = "**/*" if recursive else "*"
    files = [
        p for p in dirpath.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    results = []
    for fp in sorted(files):
        result = await ingest_file(fp)
        results.append(result)
    return results
