"""
Factory for creating and configuring LLM instances.
"""
import logging
from typing import Any, Dict, Optional, Union

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


class LLMFactory:
    """Factory class for creating LLM instances based on provider configuration."""

    def __init__(
        self,
        llm_provider: str,
        llm_model_name: str,
        temperature: float,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the LLM factory.

        Args:
            llm_provider: The LLM provider ('openai', 'ollama', or 'google')
            llm_model_name: The model name to use
            temperature: The temperature parameter for generation
            openai_api_key: OpenAI API key (required for OpenAI provider)
            google_api_key: Google API key (required for Google provider)
            logger: Logger instance for logging
        """
        self.llm_provider = llm_provider
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.logger = logger or logging.getLogger(__name__)

    def create_llm(self, use_json_format: bool = False) -> Union[ChatOpenAI, ChatOllama, ChatGoogleGenerativeAI]:
        """
        Create an LLM instance based on the configured provider.

        Args:
            use_json_format: Whether to enable JSON output format

        Returns:
            An initialized LLM instance

        Raises:
            ValueError: If the provider is unsupported or configuration is invalid
            RuntimeError: If LLM initialization fails
        """
        try:
            provider = self.llm_provider.lower()

            if provider == "openai":
                return self._create_openai_llm(use_json_format)
            if provider == "ollama":
                return self._create_ollama_llm(use_json_format)
            if provider == "google":
                return self._create_google_llm(use_json_format)

            error_msg = f"Unsupported LLM provider configured: '{self.llm_provider}'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Critical error initializing LLM (Provider: {self.llm_provider}, JSON: {use_json_format}): {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_openai_llm(self, use_json: bool) -> ChatOpenAI:
        """Create an OpenAI LLM instance."""
        try:
            self._validate_openai_api_key()

            # Build the configuration dictionary for the OpenAI LLM
            openai_config = self._build_openai_config(use_json)

            openai_llm = ChatOpenAI(**openai_config)
            self.logger.info(f"OpenAI LLM '{self.llm_model_name}' initialized successfully (JSON: {use_json}).")
            return openai_llm
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI LLM '{self.llm_model_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_ollama_llm(self, use_json: bool) -> ChatOllama:
        """Create an Ollama LLM instance."""
        try:
            format_type = self._get_ollama_format(use_json)
            ollama_llm = ChatOllama(
                model=self.llm_model_name,
                temperature=self.temperature,
                format=format_type,
            )
            self.logger.info(
                f"Ollama LLM '{self.llm_model_name}' initialized successfully (JSON: {use_json}, Format: {format_type})."
            )
            return ollama_llm
        except Exception as e:
            error_msg = f"Failed to initialize Ollama LLM '{self.llm_model_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _create_google_llm(self, use_json: bool) -> ChatGoogleGenerativeAI:
        """Create a Google Generative AI LLM instance."""
        try:
            self._validate_google_api_key()

            # Build the configuration dictionary for the Google LLM
            google_config = self._build_google_config(use_json)

            google_llm = ChatGoogleGenerativeAI(**google_config)
            self.logger.info(f"Google LLM '{self.llm_model_name}' initialized successfully (JSON: {use_json}).")
            return google_llm
        except Exception as e:
            error_msg = f"Failed to initialize Google LLM '{self.llm_model_name}': {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _validate_openai_api_key(self) -> None:
        """Validate that the OpenAI API key is present."""
        try:
            if not self.openai_api_key:
                error_msg = "OPENAI_API_KEY is missing or empty. It is required for OpenAI LLM/embedding operations."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug("OpenAI API key validation successful.")
        except Exception as e:
            error_msg = f"Unexpected error during OpenAI API key validation: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _validate_google_api_key(self) -> None:
        """Validate that the Google API key is present."""
        try:
            if not self.google_api_key:
                error_msg = "GOOGLE_API_KEY is missing or empty. It is required for Google LLM/embedding operations."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self.logger.debug("Google API key validation successful.")
        except Exception as e:
            error_msg = f"Unexpected error during Google API key validation: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _build_openai_config(self, use_json: bool) -> Dict[str, Any]:
        """Build configuration dictionary for OpenAI LLM."""
        try:
            config_args: Dict[str, Any] = {
                "model": self.llm_model_name,
                "temperature": self.temperature,
                "openai_api_key": self.openai_api_key,
            }

            # Add JSON formatting configuration if requested
            if use_json:
                config_args["model_kwargs"] = {"response_format": {"type": "json_object"}}
                self.logger.info(f"JSON response format enabled for OpenAI model '{self.llm_model_name}'.")

            self.logger.debug(f"OpenAI config built successfully: {config_args}")
            return config_args

        except Exception as e:
            error_msg = f"Failed to build OpenAI configuration: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _build_google_config(self, use_json: bool) -> Dict[str, Any]:
        """Build configuration dictionary for Google LLM."""
        try:
            # Initialize the base configuration dictionary
            config_kwargs: Dict[str, Any] = {
                "model": self.llm_model_name,
                "temperature": self.temperature,
                "google_api_key": self.google_api_key,
                "convert_system_message_to_human": True,
            }

            # Add JSON formatting configuration if requested
            if use_json:
                config_kwargs["model_kwargs"] = {"response_mime_type": "application/json"}
                self.logger.info(f"JSON response format enabled for Google model '{self.llm_model_name}'.")

            self.logger.debug(f"Google config built successfully: {config_kwargs}")
            return config_kwargs

        except Exception as e:
            error_msg = f"Failed to build Google configuration: {e}"
            self.logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _get_ollama_format(self, use_json: bool) -> Optional[str]:
        """Determine the format parameter for Ollama LLM."""
        try:
            # Determine the format string based on the use_json flag
            format_param = "json" if use_json else None
            self.logger.debug(f"Ollama format parameter determined: {format_param} (use_json: {use_json})")
            return format_param
        except Exception as e:
            error_msg = f"Unexpected error determining Ollama format parameter: {e}"
            self.logger.error(error_msg, exc_info=True)
            return None
