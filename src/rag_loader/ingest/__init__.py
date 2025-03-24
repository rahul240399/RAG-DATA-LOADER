"""Document ingestion: loading and chunking source documents."""

from rag_loader.ingest.loader import load_pdf
from rag_loader.ingest.splitter import build_splitter, split_documents

__all__ = ["build_splitter", "load_pdf", "split_documents"]
