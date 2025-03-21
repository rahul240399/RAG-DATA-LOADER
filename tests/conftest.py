"""Shared pytest fixtures."""

from pathlib import Path

import fitz  # PyMuPDF
import pytest


@pytest.fixture
def sample_pdf(tmp_path: Path) -> tuple[Path, list[str]]:
    """Create a small multi-page PDF and return its path and per-page text."""
    pages_text = [
        "Hello from page one. Retrieval augmented generation.",
        "Second page content about embeddings and vectors.",
        "Third and final page discussing chunking strategies.",
    ]
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path, pages_text
