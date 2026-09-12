"""Streamlit Community Cloud entrypoint: streamlit_app/app.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from cloudrag.cloud_session import CloudSession


st.set_page_config(page_title="CloudRAG | Ask your documents", page_icon="☁️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_generator(answer_format="source-quotes-v1"):
    from cloudrag.cloud_model import load_generator
    return load_generator()


if "documents" not in st.session_state:
    st.session_state.documents = CloudSession()
    st.session_state.messages = []
    st.session_state.upload_version = 0
    try:
        with st.spinner("Preparing semantic search… The first start downloads a small embedding model."):
            st.session_state.documents.load_samples()
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))

session = st.session_state.documents


def show_answer(result):
    labels = {
        "generated": "AI answer · Qwen2.5 0.5B",
        "quoted_evidence": "AI-selected source quotes · references checked against the text",
        "retrieved_excerpts": "Source passages · no AI generation",
        "insufficient_evidence": "Not enough evidence",
    }
    st.caption(labels[result["answer_kind"]])
    st.text(result["answer"])
    if result["answer_kind"] == "quoted_evidence":
        st.caption("These words come from the cited passages. Check whether they fully answer your question.")
    if result.get("warning"):
        st.warning(result["warning"])
    for source in result["sources"]:
        page = f" · page {source['page']}" if source.get("page") else ""
        with st.expander(f"[{source['citation']}] {source['filename']}{page}"):
            st.code(source["text"], language=None, wrap_lines=True)
            st.caption(f"Retrieval similarity: {source['score']:.3f} · A ranking score, not answer confidence.")
    st.caption(f"Response time: {result['latency_ms'] / 1000:.1f} seconds")


with st.sidebar:
    st.title("☁️ CloudRAG")
    st.caption("A cloud architecture learning project")
    generate_ai = st.toggle("Use AI to select source quotes", value=True, key="generate_ai")
    st.caption("The first AI question downloads a 491 MB model. Later questions reuse it. No API key is required.")
    st.subheader("Your documents")
    if st.session_state.pop("samples_reloaded", False):
        st.info("Sample policies were reloaded. Previous chat was cleared to use the current documents.")
    st.caption("Each browser session has its own collection. Upload non-sensitive learning material.")
    files = st.file_uploader("Add PDF, TXT or Markdown", type=["pdf", "txt", "md"],
                             accept_multiple_files=True, key=f"uploads_{st.session_state.upload_version}")
    if st.button("Index selected files", disabled=not files, use_container_width=True):
        for uploaded in files:
            try:
                with st.spinner("Creating vectors and indexing your document…"):
                    result = session.ingest(uploaded.name, uploaded.getvalue())
                st.success(f"{result['filename']}: {result['chunks']} passages ({result['status']}).")
                if result["replaced"]:
                    st.session_state.messages = []
                    st.info("This document was replaced. Previous chat was cleared so answers do not refer to its old version.")
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    st.caption("Up to 2 MiB per file, 10 documents and 100 passages. Scanned PDFs need text recognition first.")
    documents = session.engine.list_documents()
    for document in documents:
        with st.expander(document["filename"]):
            st.caption(f"{document['chunks']} searchable passages")
            if st.button("Remove", key=f"remove_{document['document_id']}"):
                try:
                    session.engine.delete_document(document["document_id"])
                    st.session_state.messages = []
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
    if st.button("Load sample policies", use_container_width=True,
                 help="Reload the sample policies and start a fresh chat."):
        # Reloading can replace a custom file with a bundled sample's filename.
        # Clear old answers even if a later file in the batch cannot be loaded.
        st.session_state.messages = []
        try:
            with st.spinner("Adding sample policies to the vector database…"):
                session.load_samples()
            st.session_state.samples_reloaded = True
            st.rerun()
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if st.button("Clear documents and chat", use_container_width=True):
        session.close()
        st.session_state.documents = CloudSession()
        st.session_state.messages = []
        st.session_state.upload_version += 1
        st.rerun()
    st.divider()
    st.caption("Documents and chat are temporary: refreshing, disconnecting or restarting may clear them.")

st.caption("DOCUMENTS → EMBEDDINGS → QDRANT SEARCH → SOURCE QUOTES")
st.title("Ask your cloud documents")
st.write("Explore backup, access, scaling and incident policies with answers you can check against the source.")
left, middle, right = st.columns(3)
health = session.engine.health()
left.metric("Documents in this session", health["documents"])
middle.metric("Searchable passages", health["chunks"])
right.metric("Answer mode", "AI-selected quotes" if generate_ai else "Sources only")
st.caption(f"Vector database: Qdrant · {health['vectors']} stored vectors · "
           f"{health['embedding_dimensions']} dimensions · BGE-small embeddings")

with st.expander("How this project works"):
    st.write("1. Documents are split into short passages.\n\n"
             "2. BGE-small turns each passage and your question into 384 numbers called an embedding. "
             "Qdrant compares these vectors using cosine similarity to find related passages, even when wording differs.\n\n"
             "3. With AI selection enabled, a small Qwen language model selects text relevant to your question.\n\n"
             "4. The app matches each selection to the supplied text, expands it to its original sentence, "
             "and adds the matching source reference.\n\n"
             "5. Unmatched selections are withheld; the retrieved passages remain available.")
    st.write("Each browser session has a separate Qdrant database in memory. Removing a document also removes "
             "its vectors. This free demonstration does not keep documents after the session ends or the app restarts. "
             "The embedding model is downloaded once (about 67 MB); questions and documents are processed on this app's server.")
    st.info("The included policies are fictional teaching examples. This app does not configure cloud infrastructure. "
            "Exact text matching checks where a quote came from. The model can still select irrelevant text "
            "or miss part of an answer. Verify relevance and completeness against the evidence. "
            "Ask each question in full—previous chat messages are displayed but are not sent to the model.")

st.caption("Try: How often are backups taken? · Do administrators need MFA? · Who acts as incident lead?")

if not st.session_state.messages:
    st.info("Your documents are ready. Ask a question below or add your own documents." if health["documents"] else
            "Your collection is empty. Upload a document or choose Load sample policies to begin.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.text(message["content"])
        else:
            show_answer(message["result"])

if question := st.chat_input("Ask a question about your documents…", max_chars=500):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding evidence and preparing an answer… First AI use can take a few minutes."):
                result = session.ask(question, get_generator if generate_ai else None)
            show_answer(result)
            st.session_state.messages.append({"role": "assistant", "result": result})
            st.session_state.messages = st.session_state.messages[-20:]
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
