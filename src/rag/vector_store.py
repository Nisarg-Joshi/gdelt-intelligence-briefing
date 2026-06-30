"""
Vector Store Module
-------------------
Handles embedding of GDELT event documents and storage/retrieval
via ChromaDB with sentence-transformers (all-MiniLM-L6-v2).

Designed for CPU-only environments — no GPU required.
First run will download the ~90MB MiniLM model; subsequent runs use cache.
"""

import logging
import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "./chroma_db"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the MiniLM embedding model.
    CPU-optimised, normalised embeddings for cosine similarity.
    ~90MB download on first use, cached locally after that.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    documents: list[dict],
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """
    Embed all documents and persist them to ChromaDB.
    Wipes any existing store at persist_dir before building.

    Parameters
    ----------
    documents : list[dict]
        Each dict must have 'text' (str) and 'metadata' (dict) keys.
        Produced by gdelt_loader.events_to_documents().
    persist_dir : str
        Where ChromaDB persists its files.

    Returns
    -------
    Chroma  — ready for similarity search.
    """
    if not documents:
        raise ValueError("No documents provided to build_vector_store.")

    # Wipe stale store so we don't mix old and new data
    if os.path.exists(persist_dir):
        logger.info(f"Clearing existing ChromaDB at {persist_dir}")
        shutil.rmtree(persist_dir)

    embeddings = get_embeddings()

    langchain_docs = [
        Document(page_content=d["text"], metadata=d["metadata"])
        for d in documents
    ]

    logger.info(f"Embedding {len(langchain_docs):,} documents into ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    logger.info("Vector store built successfully.")
    return vectorstore


def load_vector_store(persist_dir: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.
    Use this when re-opening the app without re-ingesting data.
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