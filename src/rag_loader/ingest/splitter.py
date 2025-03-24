"""Text splitting that turns loaded Documents into TextChunks."""

import hashlib
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from rag_loader.models.config import PipelineConfig
from rag_loader.models.text_chunk import TextChunk


def build_splitter(config: PipelineConfig) -> TextSplitter:
    """Build a text splitter for the configured chunking strategy.

    ``recursive`` splits on a hierarchy of separators measuring length in
    characters; ``token`` measures length in model tokens via tiktoken, keeping
    chunks within an embedding model's context window.
    """
    if config.chunk_strategy == "token":
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
    )


def _chunk_id(source: str, page: int, start: int, end: int) -> str:
    """Deterministic, page-aware chunk id so re-indexing upserts in place."""
    return hashlib.md5(f"{source}|p{page}|{start}|{end}".encode()).hexdigest()


def split_documents(docs: Sequence[Document], config: PipelineConfig) -> list[TextChunk]:
    """Split documents into TextChunks, preserving source and page provenance."""
    splitter = build_splitter(config)
    chunks: list[TextChunk] = []
    for piece in splitter.split_documents(list(docs)):
        content = piece.page_content
        source = str(piece.metadata.get("source", "unknown"))
        page = int(piece.metadata.get("page", 0))
        start = int(piece.metadata.get("start_index", 0))
        end = start + len(content)
        chunks.append(
            TextChunk(
                content=content,
                metadata={"source": source, "page": page},
                start_index=start,
                end_index=end,
                source_document=source,
                chunk_id=_chunk_id(source, page, start, end),
            )
        )
    return chunks
