"""Session-isolated semantic retrieval using embedded Qdrant collections.

The parser and exact-text chunker are shared with the native RAG engine. This
engine owns no SQLite document index: Qdrant stores and searches the learned
embedding vectors. Local Qdrant runs in memory, so these collections are small,
temporary, and specific to one browser session.
"""

import hashlib
import math
import os
from threading import RLock
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

# Qdrant imports its optional FastEmbed integration during import; Hugging Face
# reads these settings then, before the lazy embedding model is constructed.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from qdrant_client import QdrantClient, models

from cloudrag.config import Settings
from cloudrag.engine import NO_EVIDENCE_ANSWER, RAGEngine


MAX_UPLOAD_BYTES = 2 * 1024 * 1024
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMENSIONS = 384


def _default_embedding_factory():
    # Loading the website or inspecting an empty collection needs no download.
    from cloudrag.embeddings import get_embedder

    return get_embedder()


class VectorEngine:
    """Own one Qdrant collection and publish document changes atomically.

    Qdrant local mode does not provide a transaction across a document's points.
    At the project's 100-passage limit, building a candidate collection is cheap.
    A complete successful candidate replaces the current client under the same
    lock as its document metadata. Failed embedding or indexing leaves the old
    client and documents intact.
    """

    COLLECTION_NAME = "passages"

    def __init__(self, settings: Settings | None = None, embedding_factory=None,
                 min_score: float = 0.45):
        if (isinstance(min_score, bool) or not isinstance(min_score, (int, float))
                or not math.isfinite(min_score) or not 0 <= min_score <= 1):
            raise ValueError("min_score must be a finite number between 0 and 1.")
        self.settings = settings if settings is not None else Settings(mode="demo")
        self.min_score = float(min_score)
        self._embedding_factory = embedding_factory or _default_embedding_factory
        self._embedder = None
        self._embedding_model_name = DEFAULT_EMBEDDING_MODEL
        self._embedding_dimensions = DEFAULT_EMBEDDING_DIMENSIONS
        self._lock = RLock()
        self._closed = False
        self._documents: dict[str, dict] = {}
        self._points: list[models.PointStruct] = []
        self._client = self._build_client([])

    # Reuse the same upload validation, PDF page extraction, and exact source
    # slicing as the native application; _chunk only depends on self.settings.
    _validate_upload = staticmethod(RAGEngine._validate_upload)
    _extract = staticmethod(RAGEngine._extract)

    def _chunk(self, pages: list[tuple[int | None, str]]) -> list[dict]:
        return RAGEngine._chunk(self, pages)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This document session is closed. Refresh to start a new session.")

    @staticmethod
    def _dispose_client(client) -> None:
        try:
            client.close()
        except Exception:
            # A cleanup error must not turn a committed document change into a
            # reported failure. This client owns memory, not shared disk files.
            pass

    def _build_client(self, points: list[models.PointStruct]) -> QdrantClient:
        candidate = None
        try:
            candidate = QdrantClient(":memory:")
            candidate.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self._embedding_dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
            if points:
                candidate.upsert(
                    collection_name=self.COLLECTION_NAME, points=points, wait=True,
                )
            if candidate.count(self.COLLECTION_NAME, exact=True).count != len(points):
                raise RuntimeError("The candidate collection did not contain every passage.")
            return candidate
        except Exception as exc:
            if candidate is not None:
                self._dispose_client(candidate)
            raise RuntimeError(
                "The vector database could not update your documents. "
                "Your previous documents are unchanged; try again."
            ) from exc

    def _get_embedder(self):
        if self._embedder is None:
            try:
                provider = self._embedding_factory()
                dimensions = provider.dimension
                model_name = provider.model_name
                if (isinstance(dimensions, bool) or not isinstance(dimensions, int)
                        or dimensions < 1 or not isinstance(model_name, str)
                        or not model_name.strip()):
                    raise ValueError("Invalid embedding model metadata.")
                self._embedder = provider
                self._embedding_dimensions = dimensions
                self._embedding_model_name = model_name
            except Exception as exc:
                raise RuntimeError(
                    "Semantic search could not load its embedding model. "
                    "Your documents are unchanged; try again."
                ) from exc
        return self._embedder

    def _validate_vector(self, vector) -> list[float]:
        try:
            components = list(vector)
            if any(isinstance(value, bool) for value in components):
                raise ValueError("Boolean values are not embedding components.")
            values = [float(value) for value in components]
            if len(values) != self._embedding_dimensions:
                raise ValueError("Embedding dimensions do not match the collection.")
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Embedding contains a non-finite value.")
            norm = math.hypot(*values)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("Embedding has no valid direction.")
            return [value / norm for value in values]
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "The embedding model returned an invalid vector. "
                "Your documents are unchanged; try again."
            ) from exc

    def _publish(self, documents: dict[str, dict], points: list[models.PointStruct]) -> None:
        candidate = self._build_client(points)
        previous = self._client
        self._client, self._documents, self._points = candidate, documents, points
        self._dispose_client(previous)

    @staticmethod
    def _validate_capacity(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    def ingest(self, filename: str, content: bytes, max_documents: int = 10,
               max_chunks: int = 100) -> dict:
        with self._lock:
            self._ensure_open()
            if isinstance(content, bytes) and len(content) > MAX_UPLOAD_BYTES:
                raise ValueError("Choose a document smaller than 2 MiB for this free demonstration.")
            filename = self._validate_upload(filename, content)
            content_hash = hashlib.sha256(content).hexdigest()
            existing = next((document for document in self._documents.values()
                             if document["filename"] == filename), None)
            duplicate = next((document for document in self._documents.values()
                              if document["content_hash"] == content_hash), None)
            # Unchanged uploads work even when the session has reached capacity.
            if duplicate is not None:
                if existing is not None and existing["document_id"] != duplicate["document_id"]:
                    raise ValueError(
                        "This replacement duplicates a different indexed document. "
                        "Delete the old document explicitly before using the existing copy."
                    )
                return {key: duplicate[key] for key in ("document_id", "filename", "chunks")} | {
                    "status": "unchanged", "replaced": False,
                }
            self._validate_capacity(max_documents, "max_documents")
            self._validate_capacity(max_chunks, "max_chunks")
            if existing is None and len(self._documents) >= max_documents:
                raise ValueError(
                    f"This session holds up to {max_documents} documents. "
                    "Remove one before adding another."
                )
            chunks = self._chunk(self._extract(filename, content))
            replaced_chunks = existing["chunks"] if existing is not None else 0
            if len(self._points) - replaced_chunks + len(chunks) > max_chunks:
                raise ValueError(
                    f"This free demonstration holds {max_chunks} text passages per session. "
                    "Use a shorter document."
                )
            provider = self._get_embedder()
            try:
                vectors = list(provider.embed_passages([chunk["text"] for chunk in chunks]))
                if len(vectors) != len(chunks):
                    raise ValueError("Embedding count does not match the passages.")
                vectors = [self._validate_vector(vector) for vector in vectors]
            except Exception as exc:
                raise RuntimeError(
                    "Semantic search could not index this document. "
                    "Your previous documents are unchanged; try again."
                ) from exc
            document_id = existing["document_id"] if existing is not None else uuid4().hex
            points = [point for point in self._points
                      if point.payload["document_id"] != document_id]
            for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors)):
                chunk_id = f"{document_id}:{ordinal + 1}"
                points.append(models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"cloudrag:{chunk_id}")),
                    vector=vector,
                    payload={
                        "document_id": document_id, "filename": filename,
                        "page": chunk["page"], "chunk_id": chunk_id,
                        "text": chunk["text"], "ordinal": ordinal,
                        "content_hash": content_hash,
                    },
                ))
            documents = dict(self._documents)
            documents[document_id] = {
                "document_id": document_id, "filename": filename,
                "chunks": len(chunks), "content_hash": content_hash,
            }
            self._publish(documents, points)
            return {"document_id": document_id, "filename": filename, "chunks": len(chunks),
                    "status": "indexed", "replaced": existing is not None}

    def ask(self, question: str, top_k: int = 3) -> dict:
        started = perf_counter()
        with self._lock:
            self._ensure_open()
            if not isinstance(question, str) or not question.strip() or len(question) > 500:
                raise ValueError("Enter a question containing 1 to 500 characters.")
            if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
                raise ValueError("top_k must be an integer between 1 and 10.")
            question = question.strip()
            sources = []
            if self._points:
                provider = self._get_embedder()
                try:
                    query = self._validate_vector(provider.embed_query(question))
                    hits = self._client.query_points(
                        collection_name=self.COLLECTION_NAME, query=query, limit=top_k,
                        with_payload=True, with_vectors=False, score_threshold=self.min_score,
                    ).points
                except Exception as exc:
                    raise RuntimeError(
                        "Semantic search is temporarily unavailable. "
                        "Your documents are still indexed; try again."
                    ) from exc
                for hit in hits:
                    if not math.isfinite(hit.score) or hit.score <= 0:
                        continue
                    row = hit.payload
                    sources.append({
                        "citation": f"S{len(sources) + 1}", "filename": row["filename"],
                        "page": row["page"], "chunk_id": row["chunk_id"], "text": row["text"],
                        "score": round(min(hit.score, 1.0), 6),
                    })
            answer, answer_kind = NO_EVIDENCE_ANSWER, "insufficient_evidence"
            if sources:
                answer = "Exact retrieved excerpts from your documents:\n\n" + "\n\n".join(
                    f"[{source['citation']}] {source['text']}" for source in sources
                )
                answer_kind = "retrieved_excerpts"
            return {"mode": "cloud_qdrant", "answer": answer, "answer_kind": answer_kind,
                    "sources": sources, "latency_ms": round((perf_counter() - started) * 1000, 2)}

    def list_documents(self) -> list[dict]:
        with self._lock:
            self._ensure_open()
            return [{key: document[key] for key in ("document_id", "filename", "chunks")}
                    for document in sorted(self._documents.values(), key=lambda doc: doc["filename"])]

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            self._ensure_open()
            if document_id not in self._documents:
                return False
            documents = {key: document for key, document in self._documents.items()
                         if key != document_id}
            points = [point for point in self._points
                      if point.payload["document_id"] != document_id]
            self._publish(documents, points)
            return True

    def health(self) -> dict:
        with self._lock:
            self._ensure_open()
            try:
                count = self._client.count(self.COLLECTION_NAME, exact=True).count
            except Exception as exc:
                raise RuntimeError("The vector database could not report its status. Try again.") from exc
            return {"backend": "Qdrant", "mode": "cloud_qdrant",
                    "embedding_model": self._embedding_model_name,
                    "embedding_dimensions": self._embedding_dimensions,
                    "documents": len(self._documents), "chunks": len(self._points), "vectors": count}

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._dispose_client(self._client)
                self._documents.clear()
                self._points.clear()
