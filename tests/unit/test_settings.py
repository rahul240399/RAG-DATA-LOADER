"""Unit tests for environment-driven Settings."""

import pytest

from rag_loader.models.config import PipelineConfig
from rag_loader.settings import Settings


def test_defaults_match_pipeline_defaults():
    settings = Settings()
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.chromadb_collection == "rag_documents"


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "256")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "custom/model")
    settings = Settings()
    assert settings.chunk_size == 256
    assert settings.embedding_model == "custom/model"


def test_pipeline_config_builder_returns_validated_config():
    settings = Settings(chunk_size=512, chunk_overlap=64)
    config = settings.pipeline_config()
    assert isinstance(config, PipelineConfig)
    assert config.chunk_size == 512
    assert config.chunk_overlap == 64


def test_dotenv_file_is_loaded(tmp_path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / ".env").write_text("RAG_CHROMADB_COLLECTION=docs_v2\n")
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.chromadb_collection == "docs_v2"
