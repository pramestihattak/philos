import chromadb
import structlog

from src.config import settings
from src.schemas import SourceChunk

log = structlog.get_logger(__name__)

COLLECTION_NAME = "philos_docs"


class RetrieverService:
    def __init__(self) -> None:
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    def init(self) -> None:
        """Initialize ChromaDB client and collection. Call once at startup."""
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "ChromaDB initialized",
            persist_dir=settings.chroma_persist_dir,
            collection=COLLECTION_NAME,
            count=self._collection.count(),
        )

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            raise RuntimeError("RetrieverService not initialized — call init() first")
        return self._collection

    def add(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Store chunks in ChromaDB."""
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        log.debug("added chunks", count=len(chunk_ids))

    def search(self, query_embedding: list[float], top_k: int) -> list[SourceChunk]:
        """Similarity search. Returns top-k SourceChunk objects."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self.collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[SourceChunk] = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]
        ids = results.get("ids") or [[]]

        for doc, meta, dist, chunk_id in zip(docs[0], metas[0], dists[0], ids[0]):
            # cosine distance → similarity score (1 - distance)
            score = round(1.0 - float(dist), 4)
            chunks.append(
                SourceChunk(
                    doc_id=meta.get("doc_id", ""),
                    filename=meta.get("filename", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc,
                    score=score,
                )
            )
        return chunks

    def document_exists(self, doc_id: str) -> bool:
        """Return True if any chunk with this doc_id is already stored."""
        results = self.collection.get(where={"doc_id": doc_id}, limit=1)
        return bool(results["ids"])

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a doc_id. Returns number of chunks deleted."""
        results = self.collection.get(where={"doc_id": doc_id})
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
        log.info("deleted document", doc_id=doc_id, chunks=len(ids))
        return len(ids)

    def list_documents(self) -> list[dict]:
        """Return a deduplicated list of ingested documents."""
        results = self.collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in results.get("metadatas") or []:
            doc_id = meta.get("doc_id", "")
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", ""),
                    "total_chunks": 0,
                }
            seen[doc_id]["total_chunks"] += 1
        return list(seen.values())

    def is_healthy(self) -> bool:
        try:
            _ = self.collection.count()
            return True
        except Exception:
            return False


retriever = RetrieverService()
