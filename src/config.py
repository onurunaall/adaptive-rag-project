from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal
from pathlib import Path


class APISettings(BaseSettings):
    """API Key Configurations"""

    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class LLMSettings(BaseSettings):
    """LLM Provider and Model Configurations"""

    llm_provider: Literal["openai", "ollama", "google"] = "openai"
    llm_model_name: str = "gpt-4o-mini"

    openai_llm_model_name: Optional[str] = None
    ollama_llm_model_name: Optional[str] = None
    google_llm_model_name: Optional[str] = None

    temperature: float = 0.0

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    def get_model_name_for_provider(self, provider: str) -> str:
        provider = provider.lower()
        if provider == "openai" and self.openai_llm_model_name:
            return self.openai_llm_model_name
        if provider == "ollama" and self.ollama_llm_model_name:
            return self.ollama_llm_model_name
        if provider == "google" and self.google_llm_model_name:
            return self.google_llm_model_name
        return self.llm_model_name


class EmbeddingSettings(BaseSettings):
    """Embedding Provider and Model Configurations"""

    embedding_provider: Literal["openai", "gpt4all", "google"] = "openai"
    embedding_model_name: Optional[str] = None

    openai_embedding_model_name: str = "text-embedding-3-small"
    google_embedding_model_name: str = "models/embedding-001"

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    def get_model_name_for_provider(self, provider: str) -> Optional[str]:
        provider = provider.lower()
        if provider == "openai":
            return self.embedding_model_name or self.openai_embedding_model_name
        if provider == "google":
            return self.embedding_model_name or self.google_embedding_model_name
        if provider == "gpt4all":
            return self.embedding_model_name
        return self.embedding_model_name


class EngineSettings(BaseSettings):
    """CoreRAGEngine Operational Settings"""

    chunk_size: int = 500
    chunk_overlap: int = 100
    default_collection_name: str = "insight_engine_default"
    persist_directory_base: Optional[str] = None
    max_rewrite_retries: int = 1
    max_grounding_attempts: int = 5
    default_retrieval_top_k: int = 5
    chunking_strategy: str = "adaptive"  # "adaptive", "recursive", "semantic", "hybrid"
    enable_document_type_detection: bool = True
    semantic_chunking_threshold: str = "percentile"  # "percentile", "standard_deviation", "interquartile"
    code_chunk_overlap: int = 50
    academic_chunk_size: int = 800
    financial_chunk_size: int = 1000
    enable_hybrid_search: bool = False
    hybrid_search_alpha: float = 0.7  # Weight for semantic vs keyword search
    enable_advanced_grounding: bool = False
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cache_size_mb: float = 500.0  # 500 MB
    warn_large_collection_threshold: int = 10000  # Warn if >10k docs
    max_hybrid_search_documents: int = 50000  # Limit for BM25 performance
    enable_streaming_for_large_collections: bool = True

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class AgentSettings(BaseSettings):
    """AgentLoopWorkflow Settings"""

    agent_model_name: str = "gpt-4o"
    enable_tavily_search_by_default: bool = True
    enable_python_repl_by_default: bool = False

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class MCPSettings(BaseSettings):
    """MCP Server Configurations with Absolute Paths"""

    # MCP Feature Flag
    enable_mcp: bool = False  # Disabled by default for backward compatibility

    filesystem_command: str = "python"
    memory_command: str = "python"
    sql_command: str = "python"

    filesystem_transport: str = "stdio"
    memory_transport: str = "stdio"
    sql_transport: str = "stdio"

    model_config = SettingsConfigDict(env_prefix="MCP_", extra="ignore")

    @property
    def filesystem_args(self) -> list[str]:
        """Get absolute path to filesystem server"""
        project_root = Path(__file__).parent.parent
        return [str(project_root.absolute() / "mcp" / "filesystem_server.py")]

    @property
    def memory_args(self) -> list[str]:
        """Get absolute path to memory server"""
        project_root = Path(__file__).parent.parent
        return [str(project_root.absolute() / "mcp" / "memory_server.py")]

    @property
    def sql_args(self) -> list[str]:
        """Get absolute path to SQL server"""
        project_root = Path(__file__).parent.parent
        return [str(project_root.absolute() / "mcp" / "sql_server.py")]


class AppSettings(BaseSettings):
    """Overall Application Settings"""

    api: APISettings = APISettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    engine: EngineSettings = EngineSettings()
    agent: AgentSettings = AgentSettings()
    mcp: MCPSettings = MCPSettings()

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")


settings = AppSettings()
