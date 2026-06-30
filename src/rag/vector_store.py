"""
Vector Store Module
-------------------
Handles embedding of GDELT event documents and storage/retrieval
via ChromaDB using its built-in lightweight ONNX-based MiniLM
embedding function — avoids torch/transformers entirely, keeping
memory usage low enough for free-tier cloud hosting.
"""

import logging
import os
import shutil

import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "./chroma_db"


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


def build_vector_store(
    documents: list[dict],
    persist_dir: str = DEFAULT_PERSIST_DIR,
    max_documents: int = 800,
    batch_size: int = 100,
) -> Chroma:
    """
    Embed documents and persist them to ChromaDB, in small batches to
    avoid memory spikes on resource-limited environments (e.g. free-tier
    cloud hosting). Wipes any existing store at persist_dir before building.

    Parameters
    ----------
    documents : list[dict]
        Each dict must have 'text' (str) and 'metadata' (dict) keys.
    persist_dir : str
        Where ChromaDB persists its files.
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

    # Wipe stale store so we don't mix old and new data
    if os.path.exists(persist_dir):
        logger.info(f"Clearing existing ChromaDB at {persist_dir}")
        shutil.rmtree(persist_dir)

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


def load_vector_store(persist_dir: str = DEFAULT_PERSIST_DIR) -> Chroma:
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


def vector_store_exists(persist_dir: str = DEFAULT_PERSIST_DIR) -> bool:
    """Check whether a persisted ChromaDB already exists."""
    return os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0