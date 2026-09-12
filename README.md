# CloudRAG

A free educational RAG app for cloud infrastructure documents. Upload text, Markdown or text-based PDFs, ask questions, and inspect the source passages behind each answer.

## Deploy on Streamlit Community Cloud

- Branch: `main`
- Main file path: `streamlit_app/app.py`
- Python: `3.13`
- Secrets: none required

The first AI question downloads the official Qwen2.5-0.5B-Instruct Q4_K_M model (491 MB). It runs on the host CPU through llama-cpp-python; no Ollama, Docker or paid AI API is needed. TF-IDF retrieves evidence. Qwen selects text, and the app matches it exactly to the supplied passages, expands it to its surrounding source sentence, and assigns the source reference. Unmatched selections are withheld. The app labels AI-selected quotes, source-only fallback, and insufficient evidence separately. This is extractive evidence selection, not a free-form AI summary.

## Try it

Four fictional policy documents load automatically. Ask: **How often are backups taken?** Open the cited backup policy and check the daily incremental backup at 01:00 UTC and Sunday full backup at 02:00 UTC. Try an unrelated question to check abstention.

## Limits

Exact text matching checks copied provenance only. The small model can select irrelevant text, miss part of an answer, or fail to recognize that information is absent. Sources themselves can also be wrong. Open the passages to check relevance, context and completeness. Documents and chat belong to each temporary browser session; refreshing, disconnecting, sleeping or restarting can clear them. Use non-sensitive learning material. The app holds up to 10 documents, 100 passages and 2 MiB per uploaded file. It is a classroom demonstration with limited shared CPU capacity. The sample policies describe fictional infrastructure; the app does not create that infrastructure.

## Model attribution

[Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), Qwen, Apache-2.0. The app downloads and verifies the official model rather than storing weights in this repository.
