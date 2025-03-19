"""PDF ingestion using LangChain's PyMuPDF loader."""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def _normalize_metadata(doc: Document, source_path: Path, total_pages: int) -> Document:
    """Reduce loader metadata to a consistent, downstream-friendly shape.

    PyMuPDF attaches many PDF-specific fields (format, author, creationDate, ...)
    and a zero-based page index. Downstream stages only need stable provenance,
    so metadata is replaced with a minimal, predictable schema.
    """
    raw_page = doc.metadata.get("page", 0)
    doc.metadata = {
        "source": source_path.name,
        "source_path": str(source_path),
        "page": int(raw_page) + 1,  # 1-based page numbers
        "total_pages": total_pages,
    }
    return doc


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF into one LangChain Document per page with normalized metadata.

    Each returned Document carries a consistent metadata schema: ``source`` (file
    name), ``source_path``, ``page`` (1-based), and ``total_pages``, shielding
    downstream stages from loader- and PDF-specific fields.

    Args:
        path: Path to a PDF file.

    Returns:
        A list of Documents, one per page, in page order.

    Raises:
        FileNotFoundError: If the path does not point to an existing file.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = PyMuPDFLoader(str(pdf_path)).load()
    total_pages = len(pages)
    return [_normalize_metadata(page, pdf_path, total_pages) for page in pages]
