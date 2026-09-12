"""Extract -> chunk -> index -> retrieve -> answer, with a small SQLite store."""

from collections import Counter
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sqlite3
from time import perf_counter
from uuid import uuid4

from pypdf import PdfReader

from cloudrag.config import Settings
from cloudrag.providers import INSUFFICIENT_EVIDENCE, OllamaProvider, normalize_vector


MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_EXTRACTED_CHARS = 300_000
STOP_WORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have "
    "how i if in into is it its me my of on or our should so than that the their "
    "them there these they this those to us was we were what when where which who "
    "why will with would you your".split()
)
NO_EVIDENCE_ANSWER = (
    "I could not find enough relevant information in your indexed documents. "
    "Try a more specific question or add a document that covers this topic."
)


def tokenize(text: str) -> list[str]:
    """A deliberately simple word tokenizer for the offline retrieval demonstration."""
    return [word for word in re.findall(r"[^\W_]+", text.lower()) if word not in STOP_WORDS]


class RAGEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = OllamaProvider(settings) if settings.mode == "ollama" else None
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.settings.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        expected = json.dumps({
            "schema_version": 1,
            "mode": self.settings.mode,
            "embedding_model": self.settings.embedding_model if self.provider else "tfidf-v1",
            "embedding_preprocessing": "nomic-task-prefixes-v1" if self.provider else "word-counts-v1",
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
        }, sort_keys=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT UNIQUE NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    page INTEGER,
                    text TEXT NOT NULL,
                    vector TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
            """)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES ('index_config', ?)", (expected,))
            stored = connection.execute("SELECT value FROM metadata WHERE key = 'index_config'").fetchone()[0]
            if stored != expected:
                raise ValueError(
                    "This index was created with different chunk settings or an embedding model. "
                    "Restore the original settings or choose a new RAG_DATA_DIR and ingest again."
                )

    @staticmethod
    def _validate_upload(filename: str, content: bytes) -> str:
        if not isinstance(filename, str):
            raise ValueError("A filename is required.")
        filename = filename.strip()
        if (not filename or len(filename) > 200 or "/" in filename or "\\" in filename
                or any(ord(character) < 32 for character in filename)):
            raise ValueError("Use a plain filename of at most 200 characters, without folder paths.")
        if Path(filename).suffix.lower() not in {".txt", ".md", ".pdf"}:
            raise ValueError("Supported documents are .txt, .md, and text-based .pdf files.")
        if not isinstance(content, bytes) or not content:
            raise ValueError("The uploaded file is empty.")
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("The file exceeds the 5 MiB upload limit.")
        return filename

    @staticmethod
    def _extract(filename: str, content: bytes) -> list[tuple[int | None, str]]:
        pages: list[tuple[int | None, str]] = []
        if Path(filename).suffix.lower() == ".pdf":
            try:
                reader = PdfReader(BytesIO(content))
                if reader.is_encrypted:
                    raise ValueError("Encrypted PDFs are not supported. Export an unencrypted copy.")
                total = 0
                for number, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    total += len(text)
                    if total > MAX_EXTRACTED_CHARS:
                        raise ValueError("Extracted text exceeds the 300,000 character limit.")
                    pages.append((number, text))
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError("Could not read this PDF. Use a valid, text-based PDF.") from exc
        else:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError("Text and Markdown documents must use UTF-8 encoding.") from exc
            if len(text) > MAX_EXTRACTED_CHARS:
                raise ValueError("Extracted text exceeds the 300,000 character limit.")
            if "\x00" in text:
                raise ValueError("The document contains binary data. Use a UTF-8 text file.")
            pages.append((None, text))
        if not any(text.strip() for _, text in pages):
            raise ValueError("No readable text was found. Scanned PDFs need OCR before upload.")
        return pages

    def _chunk(self, pages: list[tuple[int | None, str]]) -> list[dict]:
        chunks = []
        step = self.settings.chunk_size - self.settings.chunk_overlap
        for page, text in pages:
            words = list(re.finditer(r"\S+", text))
            for start in range(0, len(words), step):
                end = min(start + self.settings.chunk_size, len(words))
                # Keep the source's exact text (including whitespace) for checkable quotations.
                passage = text[words[start].start():words[end - 1].end()]
                if tokenize(passage):
                    chunks.append({"page": page, "text": passage})
                if end == len(words):
                    break
        if not chunks:
            raise ValueError("No searchable words were found in this document.")
        return chunks

    @staticmethod
    def _unchanged(connection, filename: str, content_hash: str) -> dict | None:
        duplicate = connection.execute("SELECT * FROM documents WHERE content_hash = ?", (content_hash,)).fetchone()
        if duplicate is None:
            return None
        target = connection.execute("SELECT document_id FROM documents WHERE filename = ?", (filename,)).fetchone()
        if target is not None and target[0] != duplicate["document_id"]:
            raise ValueError(
                "This replacement duplicates a different indexed document. "
                "Delete the old document explicitly before using the existing copy."
            )
        count = connection.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (duplicate["document_id"],)).fetchone()[0]
        return {"document_id": duplicate["document_id"], "filename": duplicate["filename"],
                "chunks": count, "status": "unchanged"}

    def ingest(self, filename: str, content: bytes) -> dict:
        filename = self._validate_upload(filename, content)
        content_hash = hashlib.sha256(content).hexdigest()
        with self._connect() as connection:
            unchanged = self._unchanged(connection, filename, content_hash)
            if unchanged:
                return unchanged
        chunks = self._chunk(self._extract(filename, content))
        if self.provider:
            vectors = self.provider.embed([chunk["text"] for chunk in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError("The embedding provider returned the wrong number of vectors.")
            vectors = [normalize_vector(vector) for vector in vectors]
            if len({len(vector) for vector in vectors}) != 1:
                raise RuntimeError("The embedding provider returned inconsistent dimensions.")
        else:
            # Store sparse word counts. IDF is computed against the current corpus when searching.
            vectors = [dict(Counter(tokenize(chunk["text"]))) for chunk in chunks]
        with self._connect() as connection:
            # The transaction starts after extraction/embedding: failed model calls leave old data intact.
            connection.execute("BEGIN IMMEDIATE")
            unchanged = self._unchanged(connection, filename, content_hash)
            if unchanged:
                return unchanged
            if self.provider:
                dimension = connection.execute("SELECT value FROM metadata WHERE key = 'embedding_dimension'").fetchone()
                if dimension is not None and int(dimension[0]) != len(vectors[0]):
                    raise ValueError("Embedding dimensions changed. Use a new RAG_DATA_DIR and ingest again.")
                connection.execute("INSERT OR IGNORE INTO metadata VALUES ('embedding_dimension', ?)", (str(len(vectors[0])),))
            existing = connection.execute("SELECT document_id FROM documents WHERE filename = ?", (filename,)).fetchone()
            document_id = existing[0] if existing else uuid4().hex
            if existing:
                connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                connection.execute("UPDATE documents SET content_hash = ? WHERE document_id = ?", (content_hash, document_id))
            else:
                connection.execute("INSERT INTO documents VALUES (?, ?, ?)", (document_id, filename, content_hash))
            connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", [
                (f"{document_id}:{i + 1}", document_id, i, chunk["page"], chunk["text"], json.dumps(vector))
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ])
        return {"document_id": document_id, "filename": filename, "chunks": len(chunks), "status": "indexed"}

    @staticmethod
    def _tfidf_scores(question: str, rows: list[dict]) -> list[float]:
        documents = [json.loads(row["vector"]) for row in rows]
        frequency = Counter(term for document in documents for term in document)
        idf = {term: math.log((1 + len(documents)) / (1 + count)) + 1 for term, count in frequency.items()}
        query = {term: count * idf[term] for term, count in Counter(tokenize(question)).items() if term in idf}
        query_norm = math.hypot(*query.values())
        if not query_norm:
            return [0.0] * len(rows)
        scores = []
        for document in documents:
            vector = {term: count * idf[term] for term, count in document.items()}
            norm = math.hypot(*vector.values())
            dot = sum(weight * vector.get(term, 0) for term, weight in query.items())
            scores.append(dot / (query_norm * norm) if norm else 0.0)
        return scores

    def ask(self, question: str, top_k: int = 3) -> dict:
        started = perf_counter()
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Enter a question containing 1 to 2,000 characters.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer between 1 and 10.")
        question = question.strip()
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(
                "SELECT c.*, d.filename FROM chunks c JOIN documents d USING (document_id) "
                "ORDER BY d.filename, c.ordinal"
            ).fetchall()]
        sources = []
        if rows:
            if self.provider:
                query_vectors = self.provider.embed([question], is_query=True)
                if len(query_vectors) != 1:
                    raise RuntimeError("The embedding provider returned an invalid query vector.")
                query = normalize_vector(query_vectors[0])
                vectors = [normalize_vector(json.loads(row["vector"])) for row in rows]
                if any(len(vector) != len(query) for vector in vectors):
                    raise ValueError("Query and stored embedding dimensions differ. Rebuild in a new RAG_DATA_DIR.")
                scores = [sum(a * b for a, b in zip(query, vector)) for vector in vectors]
            else:
                scores = self._tfidf_scores(question, rows)
            ranked = sorted(zip(scores, rows), key=lambda pair: pair[0], reverse=True)
            for score, row in ranked[:top_k]:
                if score > 0 and score >= self.settings.min_score:
                    sources.append({
                        "citation": f"S{len(sources) + 1}", "filename": row["filename"],
                        "page": row["page"], "chunk_id": row["chunk_id"], "text": row["text"],
                        "score": round(min(score, 1.0), 6),
                    })
        answer, answer_kind = NO_EVIDENCE_ANSWER, "insufficient_evidence"
        if sources and self.provider:
            generated = self.provider.generate(question, sources)
            if generated != INSUFFICIENT_EVIDENCE:
                answer, answer_kind = generated, "generated"
        elif sources:
            answer = "Demo mode: exact retrieved excerpts; no language model was used.\n\n" + "\n\n".join(
                f"[{source['citation']}] {source['text']}" for source in sources
            )
            answer_kind = "retrieved_excerpts"
        return {"mode": self.settings.mode, "answer": answer, "answer_kind": answer_kind,
                "sources": sources, "latency_ms": round((perf_counter() - started) * 1000, 2)}

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT d.document_id, d.filename, COUNT(c.chunk_id) AS chunks "
                "FROM documents d LEFT JOIN chunks c USING (document_id) "
                "GROUP BY d.document_id ORDER BY d.filename"
            ).fetchall()]

    def delete_document(self, document_id: str) -> bool:
        with self._connect() as connection:
            deleted = connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            return deleted.rowcount > 0

    def health(self) -> dict:
        with self._connect() as connection:
            counts = connection.execute(
                "SELECT (SELECT COUNT(*) FROM documents) AS documents, (SELECT COUNT(*) FROM chunks) AS chunks"
            ).fetchone()
        return {"mode": self.settings.mode, "documents": counts["documents"], "chunks": counts["chunks"]}
