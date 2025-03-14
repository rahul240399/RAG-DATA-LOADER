"""Unit tests for the settings-driven LangChain factories.

The LangChain ``init_*`` entry points are patched so the tests assert that
Settings are translated into the correct provider/model selection without
pulling in heavy provider packages.
"""

from typing import Any

import pytest

from rag_loader import factories
from rag_loader.settings import Settings


def test_build_embeddings_passes_provider_and_model(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def fake_init_embeddings(*, model: str, provider: str) -> object:
        captured["model"] = model
        captured["provider"] = provider
        return object()

    monkeypatch.setattr(factories, "init_embeddings", fake_init_embeddings)

    settings = Settings(embedding_provider="openai", embedding_model="text-embedding-3-small")
    factories.build_embeddings(settings)

    assert captured == {"model": "text-embedding-3-small", "provider": "openai"}


def test_build_chat_model_passes_provider_model_and_temperature(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(*, model: str, model_provider: str, temperature: float) -> object:
        captured["model"] = model
        captured["model_provider"] = model_provider
        captured["temperature"] = temperature
        return object()

    monkeypatch.setattr(factories, "init_chat_model", fake_init_chat_model)

    settings = Settings(llm_provider="anthropic", llm_model="claude-3-5-sonnet-latest")
    factories.build_chat_model(settings)

    assert captured == {
        "model": "claude-3-5-sonnet-latest",
        "model_provider": "anthropic",
        "temperature": 0.0,
    }
