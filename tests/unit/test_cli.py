"""Tests for the Typer CLI."""

import uuid
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from typer.testing import CliRunner

from rag_loader import cli
from rag_loader.indexing import Indexer
from rag_loader.models.config import PipelineConfig
from rag_loader.store.chroma import ChromaStore

runner = CliRunner()


@pytest.fixture
def fake_indexer(monkeypatch: pytest.MonkeyPatch) -> None:
    store = ChromaStore(
        DeterministicFakeEmbedding(size=16), collection_name=f"rag_{uuid.uuid4().hex}"
    )
    indexer = Indexer(store, PipelineConfig(chunk_size=200, chunk_overlap=20))
    monkeypatch.setattr(cli, "_build_indexer", lambda: indexer)


def test_info_command_reports_configuration():
    result = runner.invoke(cli.app, ["info"])
    assert result.exit_code == 0
    assert "Vector store:" in result.stdout


def test_index_command_reports_summary(fake_indexer: None, sample_pdf: tuple[Path, list[str]]):
    path, _ = sample_pdf
    result = runner.invoke(cli.app, ["index", str(path)])
    assert result.exit_code == 0
    assert "Indexed 1/1 documents" in result.stdout


def test_index_command_fails_on_bad_document(fake_indexer: None):
    result = runner.invoke(cli.app, ["index", "/no/such/file.pdf"])
    assert result.exit_code == 1
    assert "FAILED" in result.output
