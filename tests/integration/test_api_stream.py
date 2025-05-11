"""Tests for the streaming chat endpoint."""

import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_loader.api.app import create_app, get_chat_model, get_indexer
from rag_loader.indexing import Indexer
from rag_loader.models.config import PipelineConfig
from rag_loader.models.text_chunk import TextChunk
from rag_loader.store.chroma import ChromaStore


@pytest.fixture
def client() -> TestClient:
    store = ChromaStore(
        DeterministicFakeEmbedding(size=16), collection_name=f"rag_{uuid.uuid4().hex}"
    )
    store.add_chunks(
        [
            TextChunk(
                content="some indexed context",
                metadata={"source": "d.pdf", "page": 1},
                start_index=0,
                end_index=20,
                source_document="d.pdf",
            )
        ]
    )
    app = create_app()
    app.dependency_overrides[get_indexer] = lambda: Indexer(store, PipelineConfig())
    app.dependency_overrides[get_chat_model] = lambda: FakeListChatModel(
        responses=["Streamed answer [d.pdf:1]."]
    )
    return TestClient(app)


def test_chat_stream_emits_answer_and_done(client: TestClient):
    response = client.post("/chat/stream", json={"question": "hi"})
    assert response.status_code == 200
    assert "[DONE]" in response.text
    # Reassemble streamed tokens (the fake model streams one character per SSE event).
    tokens = [
        line.partition("data:")[2].strip()
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    streamed = "".join(token for token in tokens if token != "[DONE]")
    assert "answer" in streamed.lower()
