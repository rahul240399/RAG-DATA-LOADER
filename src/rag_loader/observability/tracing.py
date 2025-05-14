"""LangSmith tracing configuration.

LangChain emits traces for every retriever, chat model, and chain invocation
when the standard ``LANGCHAIN_*`` environment is set. This module translates the
application's settings into that environment so retrieval and generation are
traceable end to end without changing call sites.
"""

import os

from rag_loader.settings import Settings


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith tracing from settings. Returns whether it was enabled."""
    if not settings.langsmith_tracing:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    return True
