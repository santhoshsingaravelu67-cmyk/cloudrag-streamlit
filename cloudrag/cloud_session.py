"""A separate, temporary document index for each Streamlit browser session."""

from pathlib import Path
from time import perf_counter

from cloudrag.engine import NO_EVIDENCE_ANSWER
from cloudrag.vector_engine import VectorEngine
from cloudrag.providers import INSUFFICIENT_EVIDENCE


MAX_DOCUMENTS = 10
MAX_SESSION_CHUNKS = 100
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_docs"


class CloudSession:
    def __init__(self, embedding_factory=None):
        self.engine = VectorEngine(embedding_factory=embedding_factory)

    def close(self):
        self.engine.close()

    def ingest(self, filename: str, content: bytes) -> dict:
        if isinstance(content, bytes) and len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("Choose a document smaller than 2 MiB for this free demonstration.")
        return self.engine.ingest(filename, content, max_documents=MAX_DOCUMENTS,
                                  max_chunks=MAX_SESSION_CHUNKS)

    def load_samples(self):
        for path in sorted(SAMPLE_DIR.glob("*.md")):
            self.ingest(path.name, path.read_bytes())

    def ask(self, question: str, generator_factory=None) -> dict:
        if not isinstance(question, str) or not question.strip() or len(question) > 500:
            raise ValueError("Enter a question containing 1 to 500 characters.")
        started = perf_counter()
        result = self.engine.ask(question, top_k=3)
        result["mode"] = "cloud_qdrant"
        if result["sources"]:
            result["answer"] = "Matching document passages are shown below. No AI answer was generated."
            if generator_factory is not None:
                try:
                    answer = generator_factory().generate(question.strip(), result["sources"])
                    if answer == INSUFFICIENT_EVIDENCE:
                        result.update(answer=NO_EVIDENCE_ANSWER, answer_kind="insufficient_evidence")
                    else:
                        result.update(answer=answer, answer_kind="quoted_evidence", mode="cloud_qdrant_quotes")
                except RuntimeError as exc:
                    result["warning"] = str(exc)
        result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
        return result
