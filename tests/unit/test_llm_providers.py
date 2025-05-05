"""Tests that the chat model factory dispatches across providers."""

from typing import Any

import pytest

from rag_loader import factories
from rag_loader.settings import Settings


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-3-5-sonnet-latest"),
        ("openai", "gpt-4o-mini"),
        ("ollama", "llama3.1"),
    ],
)
def test_build_chat_model_dispatches(monkeypatch: pytest.MonkeyPatch, provider: str, model: str):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(*, model: str, model_provider: str, temperature: float) -> object:
        captured.update(model=model, model_provider=model_provider, temperature=temperature)
        return object()

    monkeypatch.setattr(factories, "init_chat_model", fake_init_chat_model)
    factories.build_chat_model(Settings(llm_provider=provider, llm_model=model))

    assert captured["model_provider"] == provider
    assert captured["model"] == model
