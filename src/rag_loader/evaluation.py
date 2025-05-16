"""RAG evaluation harness: run the chain over a golden dataset and score it.

A deterministic keyword-recall metric makes the suite runnable offline and in
CI. The same per-sample answers and retrieved contexts can be handed to Ragas
(faithfulness, answer relevancy, context precision/recall) when an evaluator LLM
is configured via :func:`ragas_records`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable


@dataclass(frozen=True)
class EvalSample:
    question: str
    ground_truth: str
    reference_keywords: list[str]


def load_golden_dataset(path: str | Path) -> list[EvalSample]:
    """Load the golden evaluation dataset from JSON."""
    items = json.loads(Path(path).read_text())
    return [EvalSample(**item) for item in items]


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Fraction of reference keywords present in the answer (1.0 if none required)."""
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in answer.lower())
    return hits / len(keywords)


def evaluate_chain(
    chain: Runnable[dict[str, Any], dict[str, Any]], dataset: list[EvalSample]
) -> dict[str, float]:
    """Run the RAG chain over the dataset and return aggregate offline scores."""
    recalls = [
        keyword_recall(chain.invoke({"question": s.question})["answer"], s.reference_keywords)
        for s in dataset
    ]
    return {
        "answer_keyword_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "samples": float(len(dataset)),
    }


def ragas_records(
    chain: Runnable[dict[str, Any], dict[str, Any]], dataset: list[EvalSample]
) -> list[dict[str, Any]]:
    """Build per-sample records in the shape Ragas expects (requires the eval extra)."""
    records = []
    for sample in dataset:
        result = chain.invoke({"question": sample.question})
        records.append(
            {
                "user_input": sample.question,
                "response": result["answer"],
                "reference": sample.ground_truth,
            }
        )
    return records
