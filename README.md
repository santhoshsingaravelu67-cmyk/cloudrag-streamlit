# CloudRAG

A free educational RAG app for cloud infrastructure documents. Upload text, Markdown or text-based PDFs, ask questions, and inspect the source passages behind each answer. The website uses **BGE embeddings and an embedded Qdrant vector database** for semantic retrieval, with optional AI selection of checked source quotations.

## Deploy on Streamlit Community Cloud

- Branch: `main`
- Main file path: `streamlit_app/app.py`
- Python: `3.13`
- Secrets: none required

Preparing the first document collection downloads approximately **67 MB** of quantized BGE embedding weights plus tokenizer files. FastEmbed runs this model on the host CPU and creates **384-dimensional vectors**. Qdrant stores each passage's vector with its text and source metadata in a separate in-memory collection for each browser session. A question is embedded and searched by cosine similarity. No Qdrant account, separate database server, Ollama, Docker or paid AI API is needed.

The first optional AI quote question separately downloads the official Qwen2.5-0.5B-Instruct Q4_K_M model (**491 MB**), which runs through llama-cpp-python. Qwen selects text, and the app matches it to the supplied passages, expands it to its surrounding source sentence, and assigns the source reference. Unmatched selections are withheld. The app labels AI-selected quotes, source-only fallback, and insufficient evidence separately. This is extractive evidence selection, not a free-form AI summary.

The model weights and instances are shared; users' documents, vectors and chat are separate by session. Both inference stages run inside the app server, without sending user content to an external inference API. Public model downloads require internet.

## Try it

1. Open the app and turn **Use AI to select source quotes** off to check retrieval without the larger quote-selection model. Embeddings still run.
2. Wait for four fictional policy documents to load. Expect **4 documents**, **8 searchable passages**, and the caption **Vector database: Qdrant · 8 stored vectors · 384 dimensions · BGE-small embeddings**.
3. Ask **What is the schedule for making copies of our data?** Inspect the backup passages for daily incremental backups at 01:00 UTC and Sunday full backups at 02:00 UTC. Check the evidence, not just the similarity score.
4. Upload a short fictional file, confirm that passage and vector counts increase, and ask a question using different wording. Open the source to check its content.
5. Replace that file under the same name and verify the updated text. Click **Remove** and confirm that its vectors disappear and earlier chat clears. **Clear documents and chat** should leave zero documents, passages and vectors. **Load sample policies** restores the examples.

The vector update passed **218 automated software tests** locally. A small real-model retrieval check found the expected document in the top three for **14 of 14** answerable questions, including six paraphrases. Two unrelated questions returned no passages, while two questions about missing but related facts still retrieved passages. These are local retrieval checks, not generated-answer accuracy or hosted performance guarantees.

## Limits

The retriever requests up to three passages with a default cosine score of at least **0.45**. This is a demo heuristic, not an answerability test or confidence percentage. A related passage may omit the answer entirely. Exact quote matching checks copied provenance only. Qwen can select irrelevant text, miss part of an answer, or fail to recognize that information is absent. Sources themselves can also be wrong. Open the passages to check relevance, context and completeness.

An earlier hosted Qwen request timed out after **188 seconds** and returned explicit source passages. Successful hosted AI quote selection remains unverified; adding a vector database does not fix this separate limitation. Use source-only mode for the classroom demonstration and retry AI only when host resources allow.

Documents, vectors and chat belong to each temporary browser session; refreshing, disconnecting, sleeping or restarting can clear them. This is embedded Qdrant in memory, not a managed Qdrant server or persistent cloud store. A changed document is fully embedded and indexed into a candidate collection before the session switches to it; failed replacements leave the old collection intact. Use non-sensitive learning material. The app holds up to **10 documents, 100 passage vectors and 2 MiB per uploaded file**. It is a classroom demonstration with limited shared CPU capacity. The sample policies describe fictional infrastructure; the app does not create that infrastructure.

## Model attribution

[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), BAAI, MIT; quantized ONNX files from [Qdrant's fixed model revision](https://huggingface.co/qdrant/bge-small-en-v1.5-onnx-q/tree/52398278842ec682c6f32300af41344b1c0b0bb2). The pinned stack uses Qdrant client 1.19.0, FastEmbed 0.8.0 and ONNX Runtime 1.24.3, with two CPU threads and passage batches of eight.

[Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), Qwen, Apache-2.0. The app downloads model weights from fixed revisions and verifies the Qwen file checksum; weights are not stored in this repository.
