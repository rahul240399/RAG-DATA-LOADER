"""Request and response schemas for the HTTP API."""

from pydantic import BaseModel


class IngestResponse(BaseModel):
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str
    k: int = 4


class RetrievedChunk(BaseModel):
    content: str
    source: str | None = None
    page: int | None = None


class QueryResponse(BaseModel):
    chunks: list[RetrievedChunk]


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source: str | None = None
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Citation]
