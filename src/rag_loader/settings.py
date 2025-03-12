"""Application settings loaded from the environment and an optional .env file."""

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

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store (ChromaDB)
    chromadb_collection: str = "rag_documents"
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000

    def pipeline_config(self) -> PipelineConfig:
        """Build a validated :class:`PipelineConfig` from the current settings."""
        return PipelineConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_model=self.embedding_model,
            chromadb_collection=self.chromadb_collection,
            chromadb_host=self.chromadb_host,
            chromadb_port=self.chromadb_port,
        )
