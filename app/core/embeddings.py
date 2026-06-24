from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from app.core.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER

logger = logging.getLogger(__name__)

_embedder: Embeddings | None = None


def get_embedder() -> Embeddings:
    """Return the configured LangChain embedder, creating it once and caching it.

    Provider is controlled by EMBEDDING_PROVIDER in app/core/config.py.
    """
    global _embedder
    if _embedder is None:
        logger.info("Loading embedder: provider=%s model=%s", EMBEDDING_PROVIDER, EMBEDDING_MODEL)
        if EMBEDDING_PROVIDER == "local":
            from langchain_huggingface import HuggingFaceEmbeddings
            _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        elif EMBEDDING_PROVIDER == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            _embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        elif EMBEDDING_PROVIDER == "nvidia":
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
            _embedder = NVIDIAEmbeddings(model=EMBEDDING_MODEL)
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}")
        logger.info("Embedder ready")
    return _embedder
