"""Configuration shared by the API, command line, and retrieval engine."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    mode: str = "demo"
    ollama_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 180
    chunk_overlap: int = 30
    min_score: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        if self.mode not in {"demo", "ollama"}:
            raise ValueError("RAG_MODE must be 'demo' or 'ollama'.")
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive number of words.")
        if not isinstance(self.chunk_overlap, int) or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")
        if not math.isfinite(self.min_score) or not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be a finite number between 0 and 1.")
        parsed = urlparse(self.ollama_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("OLLAMA_BASE_URL must be an HTTP(S) URL without credentials or a query.")
        if not self.chat_model.strip() or not self.embedding_model.strip():
            raise ValueError("Ollama model names cannot be empty.")

    @property
    def db_path(self) -> Path:
        # Each mode owns an index: sparse demo vectors and dense model vectors cannot mix.
        return self.data_dir / f"{self.mode}.sqlite3"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("RAG_DATA_DIR", "data")),
            mode=os.environ.get("RAG_MODE", "demo").strip().lower(),
            ollama_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            chat_model=os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b"),
            embedding_model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        )
