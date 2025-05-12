"""Application settings loaded from the environment and an optional .env file."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_loader.models.config import PipelineConfig


class Settings(BaseSettings):
    """Environment-driven configuration for the pipeline and its providers.

    Values are read from environment variables prefixed with ``RAG_`` and from a
    local ``.env`` file when present. This keeps secrets and deployment-specific
    values out of the codebase and lets the same artifact run unchanged across
    local development, CI, and production.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Chunking
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    chunk_strategy: Literal["recursive", "token"] = "recursive"

    # Embeddings (provider selected via LangChain init_embeddings)
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chat model / LLM (provider selected via LangChain init_chat_model)
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-5-sonnet-latest"
    llm_temperature: float = 0.0

    # Vector store (ChromaDB)
    chromadb_collection: str = "rag_documents"
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000

    # Vector store backend selection
    vector_store: Literal["chroma", "qdrant"] = "chroma"
    chroma_persist_directory: str | None = None
    qdrant_url: str | None = None

    # Observability
    log_level: str = "INFO"
    log_json: bool = False

    def pipeline_config(self) -> PipelineConfig:
        """Build a validated :class:`PipelineConfig` from the current settings."""
        return PipelineConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunk_strategy=self.chunk_strategy,
            embedding_model=self.embedding_model,
            chromadb_collection=self.chromadb_collection,
            chromadb_host=self.chromadb_host,
            chromadb_port=self.chromadb_port,
        )
