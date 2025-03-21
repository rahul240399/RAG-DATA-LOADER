"""Tests for the PDF loader."""

from pathlib import Path

import pytest

from rag_loader.ingest import load_pdf


def test_returns_one_document_per_page(sample_pdf: tuple[Path, list[str]]):
    path, pages_text = sample_pdf
    docs = load_pdf(path)
    assert len(docs) == len(pages_text)


def test_extracts_page_text(sample_pdf: tuple[Path, list[str]]):
    path, pages_text = sample_pdf
    docs = load_pdf(path)
    for doc, expected in zip(docs, pages_text, strict=True):
        assert expected in doc.page_content


def test_normalizes_metadata_schema(sample_pdf: tuple[Path, list[str]]):
    path, pages_text = sample_pdf
    first = load_pdf(path)[0]
    assert set(first.metadata) == {"source", "source_path", "page", "total_pages"}
    assert first.metadata["source"] == path.name
    assert first.metadata["source_path"] == str(path)
    assert first.metadata["page"] == 1
    assert first.metadata["total_pages"] == len(pages_text)


def test_pages_are_numbered_sequentially(sample_pdf: tuple[Path, list[str]]):
    path, _ = sample_pdf
    docs = load_pdf(path)
    assert [d.metadata["page"] for d in docs] == list(range(1, len(docs) + 1))


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "does-not-exist.pdf")
