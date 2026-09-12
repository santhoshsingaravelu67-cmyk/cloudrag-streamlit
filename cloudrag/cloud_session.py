"""A separate, temporary document index for each Streamlit browser session."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from cloudrag.config import Settings
from cloudrag.engine import RAGEngine, NO_EVIDENCE_ANSWER
from cloudrag.providers import INSUFFICIENT_EVIDENCE


MAX_DOCUMENTS = 10
MAX_SESSION_CHUNKS = 100
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_docs"


class CloudSession:
    def __init__(self):
        self._directory = TemporaryDirectory(prefix="cloudrag-session-")
        self.engine = RAGEngine(Settings(data_dir=Path(self._directory.name), mode="demo"))

    def close(self):
        self._directory.cleanup()

    def ingest(self, filename: str, content: bytes) -> dict:
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("Choose a document smaller than 2 MiB for this free demonstration.")
        filename = self.engine._validate_upload(filename, content)
        documents = self.engine.list_documents()
        existing = next((doc for doc in documents if doc["filename"] == filename), None)
        if not existing and len(documents) >= MAX_DOCUMENTS:
            raise ValueError("This session holds up to 10 documents. Remove one before adding another.")
        incoming = self.engine._chunk(self.engine._extract(filename, content))
        current_chunks = sum(doc["chunks"] for doc in documents)
        replaced_chunks = existing["chunks"] if existing else 0
        if current_chunks - replaced_chunks + len(incoming) > MAX_SESSION_CHUNKS:
            raise ValueError("This free demonstration holds 100 text passages per session. Use a shorter document.")
        return self.engine.ingest(filename, content)

    def load_samples(self):
        for path in sorted(SAMPLE_DIR.glob("*.md")):
            self.ingest(path.name, path.read_bytes())

    def ask(self, question: str, generator_factory=None) -> dict:
        if not isinstance(question, str) or not question.strip() or len(question) > 500:
            raise ValueError("Enter a question containing 1 to 500 characters.")
        started = perf_counter()
        result = self.engine.ask(question, top_k=3)
        result["mode"] = "cloud_tfidf"
        if result["sources"]:
            result["answer"] = "Matching document passages are shown below. No AI answer was generated."
            if generator_factory is not None:
                try:
                    answer = generator_factory().generate(question.strip(), result["sources"])
                    if answer == INSUFFICIENT_EVIDENCE:
                        result.update(answer=NO_EVIDENCE_ANSWER, answer_kind="insufficient_evidence")
                    else:
                        result.update(answer=answer, answer_kind="generated", mode="cloud_qwen")
                except RuntimeError as exc:
                    result["warning"] = str(exc)
        result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
        return result
