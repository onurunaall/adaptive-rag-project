"""
Factory for creating and configuring embedding model instances.
"""
import logging
from typing import Any, Optional, Union

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EmbeddingFactory:
    """Factory class for creating embedding model instances based on provider configuration."""

    def __init__(
        self,
        embedding_provider: str,
        embedding_model_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the embedding factory.

        Args:
            embedding_provider: The embedding provider ('openai', 'gpt4all', or 'google')
            embedding_model_name: The embedding model name to use (optional, will use defaults)
            openai_api_key: OpenAI API key (required for OpenAI provider)
            google_api_key: Google API key (required for Google provider)
            logger: Logger instance for logging
        """
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.logger = logger or logging.getLogger(__name__)

    def create_embedding_model(self) -> Union[OpenAIEmbeddings, GPT4AllEmbeddings, GoogleGenerativeAIEmbeddings]:
        """
        Create an embedding model instance based on the configured provider.

        Returns:
            An initialized embedding model instance

        Raises:
            ValueError: If the provider is unsupported or configuration is invalid
            RuntimeError: If embedding model initialization fails
        """
        try:
            provider = self.embedding_provider.lower()

            if provider == "openai":
                return self._create_openai_embeddings()
            if provider == "gpt4all":
                return self._create_gpt4all_embeddings()
            if provider == "google":
                return self._create_google_embeddings()

            error_msg = f"Unsupported embedding provider configured: '{self.embedding_provider}'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Critical error initializing embedding model (Provider: {self.embedding_provider}): {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_openai_embeddings(self) -> OpenAIEmbeddings:
        """Create an OpenAI embeddings instance."""
        try:
            self._validate_openai_embedding_key()
            model_name = self._get_openai_embedding_model()
            openai_embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=model_name)
            self.logger.info(f"OpenAI Embeddings model '{model_name}' initialized successfully.")
            return openai_embeddings
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI Embeddings model '{self.embedding_model_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_gpt4all_embeddings(self) -> GPT4AllEmbeddings:
        """Create a GPT4All embeddings instance."""
        try:
            gpt4all_embeddings = GPT4AllEmbeddings()
            self.logger.info("GPT4All Embeddings model initialized successfully.")
            return gpt4all_embeddings
        except Exception as e:
            error_msg = f"Failed to initialize GPT4All Embeddings model: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_google_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Create a Google Generative AI embeddings instance."""
        try:
            self._validate_google_embedding_key()
            model_name = self._get_google_embedding_model()
            google_embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=self.google_api_key)
            self.logger.info(f"Google Embeddings model '{model_name}' initialized successfully.")
            return google_embeddings

        except Exception as e:
            error_msg = f"Failed to initialize Google Embeddings model '{self.embedding_model_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _validate_openai_embedding_key(self) -> None:
        """Validate that the OpenAI API key is present."""
        try:
            if not self.openai_api_key:
                error_msg = "OPENAI_API_KEY is missing or empty. It is required for OpenAI embedding operations."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug("OpenAI API key validation for embeddings successful.")
        except Exception as e:
            error_msg = f"Unexpected error during OpenAI embedding API key validation: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _validate_google_embedding_key(self) -> None:
        """Validate that the Google API key is present."""
        try:
            if not self.google_api_key:
                error_msg = "GOOGLE_API_KEY is missing or empty. It is required for Google embedding operations."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug("Google API key validation for embeddings successful.")
        except Exception as e:
            error_msg = f"Unexpected error during Google embedding API key validation: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _get_openai_embedding_model(self) -> str:
        """Get the OpenAI embedding model name, using default if not specified."""
        try:
            model_name = self.embedding_model_name or "text-embedding-3-small"
            self.logger.debug(f"OpenAI embedding model determined: '{model_name}' (Configured: '{self.embedding_model_name}')")
            return model_name
        except Exception as e:
            error_msg = f"Unexpected error determining OpenAI embedding model: {e}"
            self.logger.error(error_msg, exc_info=True)
            return "text-embedding-3-small"

    def _get_google_embedding_model(self) -> str:
        """Get the Google embedding model name, using default if not specified."""
        try:
            model_name = self.embedding_model_name or "models/embedding-001"
            self.logger.debug(f"Google embedding model determined: '{model_name}' (Configured: '{self.embedding_model_name}')")
            return model_name
        except Exception as e:
            error_msg = f"Unexpected error determining Google embedding model: {e}"
            self.logger.error(error_msg, exc_info=True)
            return "models/embedding-001"
