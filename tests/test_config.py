import pytest

from src.config import LLMSettings, EmbeddingSettings


def test_llm_settings_get_model_name():
    """
    Verify that LLMSettings.get_model_name_for_provider returns:
      - Default model name when no provider-specific override is given.
      - Provider-specific override when provided.
      - Fallback to general model name if no provider-specific override is set.
    """
    # 1. Default behavior for OpenAI: should return "gpt-4o"
    settings = LLMSettings(llm_provider="openai")
    model_name = settings.get_model_name_for_provider("openai")
    assert model_name == "gpt-4o"

    # 2. When a provider-specific override exists (ollama): use that override
    settings = LLMSettings(
        llm_provider="ollama",
        llm_model_name="default-model",
        ollama_llm_model_name="special-ollama-model"
    )
    model_name = settings.get_model_name_for_provider("ollama")
    assert model_name == "special-ollama-model"

    # 3. If provider-specific override is not set, fallback to llm_model_name
    settings = LLMSettings(
        llm_provider="google",
        llm_model_name="default-model",
        openai_llm_model_name="special-openai-model"
    )
    model_name = settings.get_model_name_for_provider("google")
    assert model_name == "default-model"


def test_embedding_settings_get_model_name():
    """
    Verify that EmbeddingSettings.get_model_name_for_provider returns:
      - Default embedding model names for each provider.
      - Custom embedding_model_name override when provided.
      - Correct fallback behavior for providers without specific defaults.
    """
    # 1. Default OpenAI embedding model
    settings = EmbeddingSettings(embedding_provider="openai")
    model_name = settings.get_model_name_for_provider("openai")
    assert model_name == "text-embedding-3-small"

    # 2. Custom override for OpenAI embedding model
    settings = EmbeddingSettings(
        embedding_provider="openai",
        embedding_model_name="my-custom-embedding-model"
    )
    model_name = settings.get_model_name_for_provider("openai")
    assert model_name == "my-custom-embedding-model"

    # 3. Default Google embedding model
    settings = EmbeddingSettings(embedding_provider="google")
    model_name = settings.get_model_name_for_provider("google")
    assert model_name == "models/embedding-001"

    # 4. For gpt4all, fallback to embedding_model_name if provided
    settings = EmbeddingSettings(
        embedding_provider="gpt4all",
        embedding_model_name="custom-gpt4all-model"
    )
    model_name = settings.get_model_name_for_provider("gpt4all")
    assert model_name == "custom-gpt4all-model"
