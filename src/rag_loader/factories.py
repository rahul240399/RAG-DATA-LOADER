"""Factories that build LangChain chat models and embeddings from Settings.

These thin wrappers delegate to LangChain's ``init_chat_model`` and
``init_embeddings`` so the concrete provider (HuggingFace, OpenAI, Voyage,
Anthropic, Ollama, ...) is chosen entirely through configuration. Application
code depends on the abstract ``BaseChatModel`` / ``Embeddings`` interfaces and
never imports a provider directly.
"""

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag_loader.settings import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    """Build the embeddings provider selected by ``settings``."""
    return init_embeddings(
        model=settings.embedding_model,
        provider=settings.embedding_provider,
    )


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Build the chat model selected by ``settings``."""
    return init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=settings.llm_temperature,
    )
