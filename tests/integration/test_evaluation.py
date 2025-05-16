"""Tests for the evaluation harness."""

import uuid

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_loader.evaluation import (
    EvalSample,
    evaluate_chain,
    keyword_recall,
    load_golden_dataset,
)
from rag_loader.generation import build_rag_chain
from rag_loader.models.text_chunk import TextChunk
from rag_loader.store.chroma import ChromaStore


def test_keyword_recall_scores():
    assert (
        keyword_recall("retrieval and generation over documents", ["retrieval", "documents"]) == 1.0
    )
    assert keyword_recall("nothing relevant here", ["retrieval"]) == 0.0
    assert keyword_recall("anything", []) == 1.0


def test_load_golden_dataset():
    samples = load_golden_dataset("eval/golden_dataset.json")
    assert samples
    assert all(isinstance(sample, EvalSample) for sample in samples)


def test_evaluate_chain_aggregates_scores():
    store = ChromaStore(
        DeterministicFakeEmbedding(size=16), collection_name=f"rag_{uuid.uuid4().hex}"
    )
    store.add_chunks(
        [
            TextChunk(
                content="retrieval augmented generation grounds answers in documents",
                metadata={"source": "d.pdf", "page": 1},
                start_index=0,
                end_index=50,
                source_document="d.pdf",
            )
        ]
    )
    llm = FakeListChatModel(responses=["retrieval and generation over documents"])
    chain = build_rag_chain(store.as_retriever(k=1), llm)
    dataset = [EvalSample(question="q", ground_truth="gt", reference_keywords=["retrieval"])]

    scores = evaluate_chain(chain, dataset)
    assert scores["samples"] == 1.0
    assert scores["answer_keyword_recall"] == 1.0
