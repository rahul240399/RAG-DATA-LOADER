"""Tests for LangSmith tracing configuration."""

import os

import pytest

from rag_loader.observability.tracing import configure_tracing
from rag_loader.settings import Settings


def test_tracing_disabled_by_default():
    assert configure_tracing(Settings(langsmith_tracing=False)) is False


def test_tracing_enabled_sets_environment(monkeypatch: pytest.MonkeyPatch):
    # Register LANGCHAIN_* with monkeypatch so they are removed after the test,
    # preventing tracing from leaking into other tests.
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "")
    monkeypatch.setenv("RAG_LANGSMITH_TRACING", "true")
    monkeypatch.setenv("RAG_LANGSMITH_PROJECT", "my-project")

    assert configure_tracing(Settings()) is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "my-project"
