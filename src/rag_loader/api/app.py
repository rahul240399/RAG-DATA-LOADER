"""FastAPI service exposing ingest, query, and chat endpoints."""

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, UploadFile
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from sse_starlette.sse import EventSourceResponse

from rag_loader.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from rag_loader.factories import build_chat_model
from rag_loader.generation import RAG_PROMPT, build_rag_chain, format_context
from rag_loader.indexing import Indexer
from rag_loader.observability import configure_logging
from rag_loader.settings import Settings


def get_settings() -> Settings:
    return Settings()


def get_indexer(settings: Annotated[Settings, Depends(get_settings)]) -> Indexer:
    return Indexer.from_settings(settings)


def get_chat_model(settings: Annotated[Settings, Depends(get_settings)]) -> BaseChatModel:
    return build_chat_model(settings)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app = FastAPI(title="RAG Data Loader", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        files: list[UploadFile],
        indexer: Annotated[Indexer, Depends(get_indexer)],
    ) -> IngestResponse:
        tmpdir = Path(tempfile.mkdtemp())
        paths: list[str] = []
        for upload in files:
            dest = tmpdir / (upload.filename or "upload.pdf")
            dest.write_bytes(await upload.read())
            paths.append(str(dest))
        batch = await indexer.aindex_paths(paths)
        return IngestResponse(
            total_documents=batch.total_documents,
            successful_documents=batch.successful_documents,
            failed_documents=batch.failed_documents,
            total_chunks=batch.total_chunks_processed,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(
        request: QueryRequest,
        indexer: Annotated[Indexer, Depends(get_indexer)],
    ) -> QueryResponse:
        docs = indexer.store.similarity_search(request.question, k=request.k)
        chunks = [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source"),
                page=doc.metadata.get("page"),
            )
            for doc in docs
        ]
        return QueryResponse(chunks=chunks)

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        request: ChatRequest,
        indexer: Annotated[Indexer, Depends(get_indexer)],
        llm: Annotated[BaseChatModel, Depends(get_chat_model)],
    ) -> ChatResponse:
        chain = build_rag_chain(indexer.store.as_retriever(), llm)
        result = chain.invoke({"question": request.question})
        sources = [Citation(source=c.get("source"), page=c.get("page")) for c in result["sources"]]
        return ChatResponse(answer=result["answer"], sources=sources)

    @app.post("/chat/stream")
    async def chat_stream(
        request: ChatRequest,
        indexer: Annotated[Indexer, Depends(get_indexer)],
        llm: Annotated[BaseChatModel, Depends(get_chat_model)],
    ) -> EventSourceResponse:
        docs = indexer.store.similarity_search(request.question, k=4)
        context = format_context(docs)
        chain = RAG_PROMPT | llm | StrOutputParser()

        async def token_stream() -> AsyncIterator[dict[str, str]]:
            async for token in chain.astream({"context": context, "question": request.question}):
                yield {"event": "token", "data": token}
            yield {"event": "done", "data": "[DONE]"}

        return EventSourceResponse(token_stream())

    return app


app = create_app()
