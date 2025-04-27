"""Command-line interface for the RAG data loader."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from rag_loader.indexing import Indexer
from rag_loader.settings import Settings

app = typer.Typer(
    help="Index PDFs into a vector store for retrieval.",
    no_args_is_help=True,
)


def _build_indexer() -> Indexer:
    """Build the configured indexer (indirection point for testing)."""
    return Indexer.from_settings(Settings())


@app.command()
def index(
    paths: Annotated[list[Path], typer.Argument(help="PDF files to index.")],
) -> None:
    """Index one or more PDFs and print a summary."""
    batch = asyncio.run(_build_indexer().aindex_paths(paths))
    stats = batch.get_summary_statistics()
    typer.echo(f"Indexed {stats['successful_documents']}/{stats['total_documents']} documents")
    typer.echo(f"Chunks stored: {stats['total_chunks_processed']}")
    typer.echo(f"Success rate: {stats['success_rate']:.0%}")
    for result in batch.get_failed_documents():
        typer.echo(f"  FAILED {result.source_document}: {'; '.join(result.errors)}", err=True)
    if batch.failed_documents:
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Show the resolved configuration."""
    settings = Settings()
    typer.echo(f"Embedding: {settings.embedding_provider}:{settings.embedding_model}")
    typer.echo(f"Vector store: {settings.vector_store}")
    typer.echo(f"Collection: {settings.chromadb_collection}")


if __name__ == "__main__":
    app()
