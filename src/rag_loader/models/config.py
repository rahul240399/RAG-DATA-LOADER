"""Validated configuration for the RAG indexing pipeline."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PipelineConfig(BaseModel):
    """Validated parameters for chunking, embedding, and vector storage.

    Validation runs at construction time: invalid parameters raise
    ``pydantic.ValidationError`` instead of being silently accepted, so
    misconfiguration fails fast at the edge of the system rather than surfacing
    deep inside the pipeline. Instances are immutable, making the config safe to
    share across concurrent document processing.

    Attributes:
        chunk_size: Maximum size of a text chunk in characters (> 0).
        chunk_overlap: Overlap between consecutive chunks (>= 0, < chunk_size).
        embedding_model: Name or path of the embedding model.
        chromadb_collection: Target ChromaDB collection name.
        chromadb_host: ChromaDB server host.
        chromadb_port: ChromaDB server port (1-65535).
    """

    model_config = ConfigDict(frozen=True)

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    chunk_strategy: Literal["recursive", "token"] = "recursive"
    embedding_model: NonEmptyStr = "sentence-transformers/all-MiniLM-L6-v2"
    chromadb_collection: NonEmptyStr = "rag_documents"
    chromadb_host: NonEmptyStr = "localhost"
    chromadb_port: int = Field(default=8000, ge=1, le=65535)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "PipelineConfig":
        """Ensure chunk overlap is strictly smaller than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @classmethod
    def get_defaults(cls) -> "PipelineConfig":
        """Return a PipelineConfig populated entirely with default values."""
        return cls()
