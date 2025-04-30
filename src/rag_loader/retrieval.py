"""Second-stage reranking on top of a base retriever."""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


class RerankingRetriever(BaseRetriever):
    """Retrieve a broad candidate set, then rerank/compress it.

    A custom, LCEL-compatible retriever: it fetches with ``base_retriever`` and
    reorders or trims the results with ``compressor`` (typically a cross-encoder),
    giving higher precision than first-stage similarity search alone.
    """

    base_retriever: BaseRetriever
    compressor: BaseDocumentCompressor

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.base_retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}
        )
        return list(self.compressor.compress_documents(candidates, query))


class CrossEncoderCompressor(BaseDocumentCompressor):
    """Rerank documents by cross-encoder relevance score, keeping the top ``top_n``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    top_n: int = 4

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        docs = list(documents)
        if not docs:
            return []
        scores = self.model.predict([(query, doc.page_content) for doc in docs])
        ranked = sorted(zip(docs, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]


def build_reranking_retriever(
    base_retriever: BaseRetriever, compressor: BaseDocumentCompressor
) -> RerankingRetriever:
    """Wrap a base retriever so its results are reranked by ``compressor``."""
    return RerankingRetriever(base_retriever=base_retriever, compressor=compressor)


def cross_encoder_compressor(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 4
) -> CrossEncoderCompressor:
    """Build a cross-encoder reranker (requires the ``huggingface`` extra)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoderCompressor(model=CrossEncoder(model_name), top_n=top_n)
