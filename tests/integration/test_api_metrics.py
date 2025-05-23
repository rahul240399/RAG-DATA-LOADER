"""Tests for the readiness and Prometheus metrics endpoints."""

import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_loader.api.app import create_app, get_indexer
from rag_loader.indexing import Indexer
from rag_loader.models.config import PipelineConfig
from rag_loader.store.chroma import ChromaStore


@pytest.fixture
def client() -> TestClient:
    indexer = Indexer(
        ChromaStore(DeterministicFakeEmbedding(size=16), collection_name=f"rag_{uuid.uuid4().hex}"),
        PipelineConfig(),
    )
    app = create_app()
    app.dependency_overrides[get_indexer] = lambda: indexer
    return TestClient(app)


def test_ready_endpoint(client: TestClient):
    assert client.get("/ready").json() == {"status": "ready"}


def test_metrics_endpoint_exposes_counters(client: TestClient):
    client.get("/health")  # generate at least one request
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_requests_total" in response.text
    assert "rag_request_duration_seconds" in response.text
