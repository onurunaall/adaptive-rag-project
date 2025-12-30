"""
RAG (Retrieval-Augmented Generation) package.

This package contains modular components for building RAG systems.
"""

from src.rag.models import (
    GroundingCheck,
    RelevanceGrade,
    RerankScore,
    QueryAnalysis,
    CoreGraphState,
)
from src.rag.llm_factory import LLMFactory
from src.rag.embedding_factory import EmbeddingFactory
from src.rag.text_splitter_factory import TextSplitterFactory
from src.rag.chain_factory import ChainFactory
from src.rag.error_handler import ErrorHandler
from src.rag.cache_manager import CacheManager
from src.rag.document_manager import DocumentManager
from src.rag.vector_store_manager import VectorStoreManager

__all__ = [
    # Models
    "GroundingCheck",
    "RelevanceGrade",
    "RerankScore",
    "QueryAnalysis",
    "CoreGraphState",
    # Factories
    "LLMFactory",
    "EmbeddingFactory",
    "TextSplitterFactory",
    "ChainFactory",
    # Managers
    "DocumentManager",
    "VectorStoreManager",
    # Utilities
    "ErrorHandler",
    "CacheManager",
]
