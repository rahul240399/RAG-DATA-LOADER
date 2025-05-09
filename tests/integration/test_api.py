"""Tests for the FastAPI service with injected in-memory dependencies."""

import uuid
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_loader.api.app import create_app, get_chat_model, get_indexer
from rag_loader.indexing import Indexer
from rag_loader.models.config import PipelineConfig
from rag_loader.store.chroma import ChromaStore


@pytest.fixture
def client() -> TestClient:
    indexer = Indexer(
        ChromaStore(DeterministicFakeEmbedding(size=16), collection_name=f"rag_{uuid.uuid4().hex}"),
        PipelineConfig(chunk_size=200, chunk_overlap=20),
    )
    app = create_app()
    app.dependency_overrides[get_indexer] = lambda: indexer
    app.dependency_overrides[get_chat_model] = lambda: FakeListChatModel(
        responses=["The answer is here [doc.pdf:1]."]
    )
    return TestClient(app)


def _make_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "retrieval augmented generation content for testing")
    doc.save(str(path))
    doc.close()
    return path


def test_health(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_query_and_chat(client: TestClient, tmp_path: Path):
    pdf = _make_pdf(tmp_path / "doc.pdf")
    with pdf.open("rb") as handle:
        ingest = client.post("/ingest", files={"files": ("doc.pdf", handle, "application/pdf")})
    assert ingest.status_code == 200
    assert ingest.json()["successful_documents"] == 1

    query = client.post("/query", json={"question": "retrieval", "k": 2})
    assert query.status_code == 200
    assert query.json()["chunks"]

    chat = client.post("/chat", json={"question": "what is this about?"})
    assert chat.status_code == 200
    assert chat.json()["answer"] == "The answer is here [doc.pdf:1]."
