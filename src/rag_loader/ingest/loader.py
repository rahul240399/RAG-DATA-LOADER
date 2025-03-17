"""PDF ingestion using LangChain's PyMuPDF loader."""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF into one LangChain Document per page.

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
    loader = PyMuPDFLoader(str(pdf_path))
    return loader.load()
