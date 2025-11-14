"""
Factory for creating and configuring text splitter instances.
"""
import logging
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from src.chunking import AdaptiveChunker, BaseChunker, HybridChunker
from src.config import settings as app_settings

try:
    import tiktoken
except ImportError:
    tiktoken = None


class TextSplitterFactory:
    """Factory class for creating text splitter instances based on chunking strategy."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        llm_provider: str,
        llm_model_name: str,
        openai_api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the text splitter factory.

        Args:
            chunk_size: The target chunk size
            chunk_overlap: The overlap between chunks
            llm_provider: The LLM provider (used for tiktoken optimization)
            llm_model_name: The LLM model name (used for tiktoken)
            openai_api_key: OpenAI API key (required for semantic and adaptive strategies)
            logger: Logger instance for logging
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_provider = llm_provider
        self.llm_model_name = llm_model_name
        self.openai_api_key = openai_api_key
        self.logger = logger or logging.getLogger(__name__)

    def create_text_splitter(self, strategy: Optional[str] = None) -> BaseChunker:
        """
        Create a text splitter based on the specified or configured strategy.

        Args:
            strategy: The chunking strategy to use. If None, uses the config default.
                     Options: 'adaptive', 'semantic', 'hybrid', or falls back to default.

        Returns:
            A text splitter instance

        Raises:
            RuntimeError: If text splitter initialization fails
        """
        try:
            # Get the configured chunking strategy, defaulting to 'adaptive'
            if strategy is None:
                strategy = getattr(app_settings.engine, "chunking_strategy", "adaptive")

            self.logger.debug(f"Initializing text splitter with strategy: '{strategy}'")

            if strategy == "adaptive":
                return self._create_adaptive_splitter()
            elif strategy == "semantic" and self.openai_api_key:
                return self._create_semantic_splitter()
            elif strategy == "hybrid":
                return self._create_hybrid_splitter()
            else:
                # Fallback to default splitter for unknown strategies or missing API key for semantic
                self.logger.info(f"Falling back to default splitter for strategy: '{strategy}'")
                return self._create_default_splitter()

        except Exception as e:
            error_msg = f"Critical error initializing text splitter (Strategy: {strategy}): {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_adaptive_splitter(self) -> AdaptiveChunker:
        """
        Create adaptive chunker with configuration.

        This method instantiates an AdaptiveChunker with the configured
        chunk size, overlap, and API key.

        Returns:
            AdaptiveChunker: An instance of the AdaptiveChunker.
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

        This method attempts to instantiate a SemanticChunker using OpenAI embeddings.
        If the required dependencies are not available, it falls back to creating
        an AdaptiveChunker.

        Returns:
            BaseChunker: An instance of the SemanticChunker or a fallback AdaptiveChunker.
        """
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            # Get the threshold type from config, defaulting to 'percentile'
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
            # Fallback to AdaptiveChunker
            return self._create_adaptive_splitter()

    def _create_hybrid_splitter(self) -> HybridChunker:
        """
        Create hybrid chunker with primary and secondary splitters.

        This method creates a HybridChunker that uses an AdaptiveChunker as the primary
        splitter and a RecursiveCharacterTextSplitter as the secondary splitter.

        Returns:
            HybridChunker: An instance of the HybridChunker.
        """
        try:
            # Create the primary splitter (AdaptiveChunker)
            primary_splitter = self._create_adaptive_splitter()

            # Create the secondary splitter (RecursiveCharacterTextSplitter with smaller chunks)
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

        This method creates a RecursiveCharacterTextSplitter. If `tiktoken` is available
        and the LLM provider is OpenAI, it attempts to create a splitter that uses
        a Tiktoken encoder for more accurate token-based splitting.

        Returns:
            RecursiveCharacterTextSplitter: An instance of the text splitter.
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

            # Create and return the default RecursiveCharacterTextSplitter
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
