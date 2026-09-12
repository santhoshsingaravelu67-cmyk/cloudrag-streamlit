"""Shared CPU embedding weights; document vectors remain in each caller's session.

Only public model files are downloaded. Documents and questions are embedded
inside this process, never submitted to a hosted inference API.
"""

import os
from pathlib import Path
import threading


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REPO = "qdrant/bge-small-en-v1.5-onnx-q"
MODEL_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"
MODEL_FILES = ["model_optimized.onnx", "config.json", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json"]
DIMENSION = 384
LOCK_TIMEOUT = 30.0
_load_lock = threading.Lock()
_embedder = None


class CPUEmbedder:
    model_name = MODEL_NAME
    dimension = DIMENSION

    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()

    def _encode(self, texts, *, query=False):
        if not self._lock.acquire(timeout=LOCK_TIMEOUT):
            raise RuntimeError("Vector search is busy. Please try again shortly.")
        try:
            encoded = (self.model.query_embed(texts[0]) if query else
                       self.model.passage_embed(texts, batch_size=8, parallel=None))
            # Materialize under the lock: FastEmbed returns a lazy iterator.
            return [vector.tolist() for vector in encoded]
        except Exception as exc:
            raise RuntimeError("Could not create document vectors. Please try again.") from exc
        finally:
            self._lock.release()

    def embed_passages(self, texts):
        return self._encode(texts)

    def embed_query(self, question):
        vectors = self._encode([question], query=True)
        if len(vectors) != 1:
            raise RuntimeError("The embedding model returned an invalid question vector.")
        return vectors[0]


def _load_embedder():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    from fastembed import TextEmbedding

    cache = Path(os.environ.get("CLOUDRAG_EMBEDDING_DIR", "~/.cache/cloudrag/embeddings")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    arguments = dict(repo_id=MODEL_REPO, revision=MODEL_REVISION, allow_patterns=MODEL_FILES,
                     cache_dir=str(cache), token=False, max_workers=1)
    try:
        directory = snapshot_download(**arguments, local_files_only=True)
        if not all((Path(directory) / name).is_file() for name in MODEL_FILES):
            raise FileNotFoundError("The model cache is incomplete.")
    except (OSError, ValueError):
        directory = snapshot_download(**arguments)
    model = TextEmbedding(model_name=MODEL_NAME, specific_model_path=directory,
                          cache_dir=str(cache), local_files_only=True, threads=2,
                          providers=["CPUExecutionProvider"])
    return CPUEmbedder(model)


def get_embedder():
    """Load once per process, retry failed loads, and never cache user text."""
    global _embedder
    if _embedder is not None:
        return _embedder
    if not _load_lock.acquire(timeout=LOCK_TIMEOUT):
        raise RuntimeError("The embedding model is starting. Please try again shortly.")
    try:
        if _embedder is None:
            try:
                _embedder = _load_embedder()
            except Exception as exc:
                raise RuntimeError(
                    "The embedding model could not start. Choose Load sample policies or retry your upload shortly."
                ) from exc
        return _embedder
    finally:
        _load_lock.release()
