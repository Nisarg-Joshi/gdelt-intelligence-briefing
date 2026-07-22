"""
Vector Store Module
-------------------
Handles embedding of GDELT event documents and storage/retrieval
via ChromaDB using its built-in lightweight ONNX-based MiniLM
embedding function — avoids torch/transformers entirely, keeping
memory usage low enough for free-tier cloud hosting.

Each build_vector_store() call uses a fresh, uniquely-named directory
rather than wiping and reusing a fixed path. Reusing a fixed path while
a previous Chroma client from an earlier run is still alive in memory
(common in Streamlit, where objects can outlive a rerun) causes SQLite
to see its underlying file deleted out from under an open connection,
which surfaces as "attempt to write a readonly database". Using a new
directory per build avoids that race entirely; old directories are
cleaned up on a best-effort basis.
"""

import logging
import os
import shutil
import uuid

import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Parent directory under which each build gets its own unique subfolder.
PERSIST_ROOT = "/tmp/chroma_db"
DEFAULT_PERSIST_DIR = PERSIST_ROOT  # kept for backwards compatibility


class ChromaDefaultEmbeddings(Embeddings):
    """
    Thin LangChain-compatible wrapper around ChromaDB's built-in
    ONNX-based MiniLM embedding function. Much lighter on memory
    than the full sentence-transformers + torch stack.
    """

    def __init__(self):
        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._fn([text])[0]


def get_embeddings() -> ChromaDefaultEmbeddings:
    """
    Load the lightweight ONNX-based MiniLM embedding function.
    No torch/transformers dependency — low memory footprint,
    suitable for resource-constrained environments.
    """
    logger.info("Loading lightweight ONNX embedding function (ChromaDB default)...")
    return ChromaDefaultEmbeddings()


def _new_persist_dir() -> str:
    """Create and return a fresh, uniquely-named persistence directory."""
    os.makedirs(PERSIST_ROOT, exist_ok=True)
    path = os.path.join(PERSIST_ROOT, f"run_{uuid.uuid4().hex[:12]}")
    os.makedirs(path, exist_ok=True)
    return path


def _cleanup_old_runs(keep_dir: str) -> None:
    """
    Best-effort cleanup of previous run directories under PERSIST_ROOT,
    so /tmp doesn't grow unbounded across many reruns in one session.
    Failures are ignored — a stale directory left behind is harmless,
    whereas failing to build the new store is not.
    """
    if not os.path.exists(PERSIST_ROOT):
        return
    for entry in os.listdir(PERSIST_ROOT):
        full_path = os.path.join(PERSIST_ROOT, entry)
        if full_path == keep_dir:
            continue
        shutil.rmtree(full_path, ignore_errors=True)


def build_vector_store(
    documents: list[dict],
    persist_dir: str | None = None,
    max_documents: int = 800,
    batch_size: int = 100,
) -> Chroma:
    """
    Embed documents and persist them to a fresh ChromaDB directory, in
    small batches to avoid memory spikes on resource-limited environments
    (e.g. free-tier cloud hosting).

    Parameters
    ----------
    documents : list[dict]
        Each dict must have 'text' (str) and 'metadata' (dict) keys.
    persist_dir : str | None
        Where ChromaDB persists its files. If None (default), a fresh
        unique directory is created automatically — this is the
        recommended usage, since it avoids the "readonly database" error
        that can occur when reusing a fixed path across reruns.
    max_documents : int
        Hard cap on number of documents embedded, to keep memory usage
        bounded on small hosting tiers.
    batch_size : int
        Number of documents embedded per batch — keeps peak memory low.

    Returns
    -------
    Chroma  — ready for similarity search.
    """
    if not documents:
        raise ValueError("No documents provided to build_vector_store.")

    if len(documents) > max_documents:
        logger.warning(
            f"Capping documents from {len(documents):,} to {max_documents:,} "
            f"to keep memory usage bounded on this environment."
        )
        documents = documents[:max_documents]

    if persist_dir is None:
        persist_dir = _new_persist_dir()
        logger.info(f"Using fresh vector store directory: {persist_dir}")
        # Clean up prior runs now that we're safely building into a new dir.
        _cleanup_old_runs(keep_dir=persist_dir)
    else:
        os.makedirs(persist_dir, exist_ok=True)

    embeddings = get_embeddings()

    langchain_docs = [
        Document(page_content=d["text"], metadata=d["metadata"])
        for d in documents
    ]

    logger.info(f"Embedding {len(langchain_docs):,} documents into ChromaDB in batches of {batch_size}...")

    vectorstore = None
    for i in range(0, len(langchain_docs), batch_size):
        batch = langchain_docs[i:i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
        else:
            vectorstore.add_documents(batch)
        logger.info(f"  -> embedded {min(i + batch_size, len(langchain_docs)):,} / {len(langchain_docs):,}")

    logger.info("Vector store built successfully.")
    return vectorstore


def load_vector_store(persist_dir: str) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.
    """
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"No ChromaDB found at '{persist_dir}'. "
            "Run build_vector_store() first."
        )
    embeddings = get_embeddings()
    logger.info(f"Loading existing vector store from {persist_dir}")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def vector_store_exists(persist_dir: str = PERSIST_ROOT) -> bool:
    """Check whether a persisted ChromaDB already exists."""
    return os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0
