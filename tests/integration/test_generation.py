"""Tests for the RAG generation chain using a fake chat model."""

import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_loader.generation import build_rag_chain, format_context
from rag_loader.models.text_chunk import TextChunk
from rag_loader.store.chroma import ChromaStore


def _chunk(content: str, start: int) -> TextChunk:
    return TextChunk(
        content=content,
        metadata={"source": "facts.pdf", "page": 1},
        start_index=start,
        end_index=start + len(content),
        source_document="facts.pdf",
    )


def test_format_context_tags_sources():
    doc = Document(
        page_content="Paris is the capital of France.",
        metadata={"source": "facts.pdf", "page": 1},
    )
    rendered = format_context([doc])
    assert "[facts.pdf:1]" in rendered
    assert "Paris" in rendered


def test_rag_chain_returns_answer_and_sources():
    store = ChromaStore(
        DeterministicFakeEmbedding(size=16), collection_name=f"rag_{uuid.uuid4().hex}"
    )
    store.add_chunks([_chunk("Paris is the capital of France.", 0)])
    llm = FakeListChatModel(responses=["The capital is Paris [facts.pdf:1]."])

    chain = build_rag_chain(store.as_retriever(k=1), llm)
    result = chain.invoke({"question": "What is the capital of France?"})

    assert result["answer"] == "The capital is Paris [facts.pdf:1]."
    assert result["sources"] == [{"source": "facts.pdf", "page": 1}]
