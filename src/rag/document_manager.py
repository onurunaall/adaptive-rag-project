"""
Document Manager Module

Handles document loading and splitting operations for the RAG engine.
Extracted from CoreRAGEngine to improve modularity and maintainability.
"""

import os
import logging
import tempfile
from typing import List, Any, Optional

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from src.config import settings as app_settings
from src.chunking import AdaptiveChunker, BaseChunker, HybridChunker

try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
except ImportError:
    UploadedFile = None

try:
    import tiktoken
except ImportError:
    tiktoken = None


class DocumentManager:
    """
    Manages document loading from various sources and splitting into chunks.

    Responsibilities:
    - Load documents from URLs, file paths, or uploaded files
    - Split documents using various chunking strategies (adaptive, semantic, hybrid, default)
    - Configure and create appropriate text splitters based on settings
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        openai_api_key: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model_name: str = "gpt-3.5-turbo",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize DocumentManager.

        Args:
            chunk_size: Maximum size of each document chunk
            chunk_overlap: Number of characters to overlap between chunks
            openai_api_key: OpenAI API key for semantic chunking
            llm_provider: LLM provider name (openai, google, ollama)
            llm_model_name: Name of the LLM model
            logger: Optional logger instance
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.llm_provider = llm_provider
        self.llm_model_name = llm_model_name
        self.logger = logger or logging.getLogger(__name__)

        self.text_splitter = self._init_text_splitter()

    def load_documents(self, source_type: str, source_value: Any) -> List[Document]:
        """
        Load documents from various source types.

        Args:
            source_type: Type of source (url, pdf_path, text_path, uploaded_pdf)
            source_value: The source value (URL string, file path, or uploaded file)

        Returns:
            List of loaded documents
        """
        try:
            if source_type == "url":
                loader = WebBaseLoader(web_paths=[source_value])
            elif source_type == "pdf_path":
                if not os.path.exists(source_value):
                    self.logger.error(f"PDF not found: {source_value}")
                    return []
                loader = PyPDFLoader(file_path=source_value)
            elif source_type == "text_path":
                if not os.path.exists(source_value):
                    self.logger.error(f"Text file not found: {source_value}")
                    return []
                loader = TextLoader(file_path=source_value)
            elif source_type == "uploaded_pdf" and UploadedFile and isinstance(source_value, UploadedFile):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(source_value.getbuffer())
                tmp.close()
                loader = PyPDFLoader(file_path=tmp.name)
            else:
                self.logger.error(f"Unsupported source_type: {source_type}")
                return []
        except Exception as e:
            self.logger.error(f"Error creating loader: {e}")
            return []

        try:
            docs = loader.load()
            self.logger.info(f"Loaded {len(docs)} docs from {source_type}")
            return docs
        except Exception as e:
            self.logger.error(f"Error loading docs: {e}")
            return []
        finally:
            if source_type == "uploaded_pdf" and "tmp" in locals():
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    def split_documents(self, docs: List[Document], strategy: Optional[str] = None) -> List[Document]:
        """
        Split documents into chunks using configured text splitter.

        Args:
            docs: List of documents to split
            strategy: Optional splitting strategy override (default, adaptive, semantic, hybrid).
                     If None, uses the instance's configured text_splitter.

        Returns:
            List of document chunks
        """
        if not docs:
            return []

        # Use configured splitter (strategy parameter is accepted for API compatibility
        # but the splitter is configured at initialization time)
        if isinstance(self.text_splitter, BaseChunker):
            chunks = self.text_splitter.chunk_documents(docs)
        else:
            chunks = self.text_splitter.split_documents(docs)

        self.logger.info(f"Split into {len(chunks)} chunks using {type(self.text_splitter).__name__}")

        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            self.logger.info(f"Average chunk size: {avg_size:.0f} characters")

        return chunks

    def _init_text_splitter(self) -> BaseChunker:
        """
        Initialize text splitter based on configured strategy.

        Returns:
            Configured text splitter instance
        """
        try:
            strategy = getattr(app_settings.engine, "chunking_strategy", "adaptive")
            self.logger.debug(f"Initializing text splitter with strategy: '{strategy}'")

            if strategy == "adaptive":
                return self._create_adaptive_splitter()
            elif strategy == "semantic" and self.openai_api_key:
                return self._create_semantic_splitter()
            elif strategy == "hybrid":
                return self._create_hybrid_splitter()
            else:
                self.logger.info(f"Falling back to default splitter for strategy: '{strategy}'")
                return self._create_default_splitter()

        except Exception as e:
            error_msg = f"Critical error initializing text splitter (Strategy: {getattr(app_settings.engine, 'chunking_strategy', 'N/A')}): {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_adaptive_splitter(self) -> AdaptiveChunker:
        """
        Create adaptive chunker with configuration.

        Returns:
            AdaptiveChunker instance
        """
        try:
            adaptive_splitter = AdaptiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                openai_api_key=self.openai_api_key,
                model_name=self.llm_model_name,
            )
            self.logger.info(
                f"AdaptiveChunker created successfully with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
            )
            return adaptive_splitter
        except Exception as e:
            error_msg = f"Failed to create AdaptiveChunker: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_semantic_splitter(self) -> BaseChunker:
        """
        Create semantic chunker with fallback to adaptive.

        Returns:
            SemanticChunker instance or fallback AdaptiveChunker
        """
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            threshold_type = getattr(app_settings.engine, "semantic_chunking_threshold", "percentile")

            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

            semantic_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type=threshold_type)
            self.logger.info(f"SemanticChunker created successfully with threshold_type='{threshold_type}'.")
            return semantic_splitter

        except ImportError as import_error:
            warning_msg = f"SemanticChunker not available (ImportError: {import_error}), falling back to AdaptiveChunker."
            self.logger.warning(warning_msg)
            return self._create_adaptive_splitter()
        except Exception as e:
            error_msg = f"Failed to create SemanticChunker, falling back to AdaptiveChunker: {e}"
            self.logger.warning(error_msg, exc_info=True)
            return self._create_adaptive_splitter()

    def _create_hybrid_splitter(self) -> HybridChunker:
        """
        Create hybrid chunker with primary and secondary splitters.

        Returns:
            HybridChunker instance
        """
        try:
            primary_splitter = self._create_adaptive_splitter()

            secondary_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size // 2,
                chunk_overlap=self.chunk_overlap // 2,
                length_function=len,
            )

            hybrid_splitter = HybridChunker(primary_splitter, secondary_splitter)
            self.logger.info(
                f"HybridChunker created successfully with primary (Adaptive) and secondary (Recursive, "
                f"chunk_size={self.chunk_size // 2}, chunk_overlap={self.chunk_overlap // 2})."
            )
            return hybrid_splitter

        except Exception as e:
            error_msg = f"Failed to create HybridChunker: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_default_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Create default recursive character splitter with tiktoken support.

        Returns:
            RecursiveCharacterTextSplitter instance
        """
        try:
            use_tiktoken = tiktoken is not None and self.llm_provider.lower() == "openai"

            if use_tiktoken:
                try:
                    tiktoken_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        model_name=self.llm_model_name,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )

                    self.logger.info(f"Default splitter created with Tiktoken encoder for model '{self.llm_model_name}'.")
                    return tiktoken_splitter
                except Exception as tiktoken_error:
                    self.logger.warning(
                        f"Tiktoken splitter failed (Error: {tiktoken_error}), falling back to default character splitter."
                    )

            default_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )

            fallback_type = "Tiktoken fallback" if use_tiktoken else "default"
            self.logger.info(
                f"Default splitter created ({fallback_type}) with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}."
            )
            return default_splitter

        except Exception as e:
            error_msg = f"Failed to create default RecursiveCharacterTextSplitter: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
