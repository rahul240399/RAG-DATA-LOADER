"""RAG answer generation: retrieve context, then synthesize a cited answer."""

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise assistant. Answer the question using ONLY the provided "
            "context. Cite supporting sources inline as [source:page]. If the context is "
            "insufficient, say you do not know.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_context(documents: Sequence[Document]) -> str:
    """Render retrieved documents as a citation-tagged context block."""
    blocks = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{source}:{page}] {doc.page_content}")
    return "\n\n".join(blocks)


def _sources(documents: Sequence[Document]) -> list[dict[str, Any]]:
    """Deduplicated source/page citations for the retrieved documents."""
    citations: list[dict[str, Any]] = []
    for doc in documents:
        citation = {"source": doc.metadata.get("source"), "page": doc.metadata.get("page")}
        if citation not in citations:
            citations.append(citation)
    return citations


def build_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    prompt: ChatPromptTemplate = RAG_PROMPT,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Build an LCEL RAG chain: ``{"question": ...} -> {"answer", "sources"}``."""
    generate = prompt | llm | StrOutputParser()
    retrieve: RunnableLambda[dict[str, Any], list[Document]] = RunnableLambda(
        lambda x: retriever.invoke(x["question"])
    )
    add_context: RunnableLambda[dict[str, Any], str] = RunnableLambda(
        lambda x: format_context(x["documents"])
    )
    finalize: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(
        lambda x: {"answer": x["answer"], "sources": _sources(x["documents"])}
    )
    return (
        RunnablePassthrough.assign(documents=retrieve)
        .assign(context=add_context)
        .assign(answer=generate)
        | finalize
    )
