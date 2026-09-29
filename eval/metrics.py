"""Pure retrieval metrics for the golden-set evaluation (spec section 10.2).

No LLM calls here - these only need the golden set's page numbers and the
retriever's output. Generation-quality metrics (faithfulness, answer relevancy)
that do need an LLM judge live in eval/llm_judge.py instead.
"""

from __future__ import annotations


def hit_rate_at_k(retrieved_pages: list[int], golden_pages: list[int]) -> bool:
    """True if any golden page appears among the retrieved pages."""
    return bool(set(retrieved_pages) & set(golden_pages))


def reciprocal_rank(retrieved_pages: list[int], golden_pages: list[int]) -> float:
    """1/rank of the first retrieved page that is a golden page, else 0.0."""
    golden = set(golden_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in golden:
            return 1.0 / rank
    return 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile of ``values`` (p in [0, 100]); 0.0 if empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(round(p / 100 * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[idx]


def mean(values: list[float]) -> float:
    """Arithmetic mean; 0.0 for an empty list (rather than raising)."""
    return sum(values) / len(values) if values else 0.0
