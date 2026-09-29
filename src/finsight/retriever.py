"""Vector retrieval: embed a question, search ChromaDB, apply metadata filters.

Vector search only (FR4, FR5). Hybrid BM25 + cross-encoder reranking is a stretch
goal (FR10, experiments E4/E5 in the requirements doc) and belongs behind its own
config flag once added - it is not part of this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from finsight.config import Config
from finsight.embeddings import Embedder
from finsight.ingest import DEFAULT_RETRIEVAL_FILTER


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk: its text, full metadata, and similarity distance."""

    text: str
    metadata: dict
    distance: float


def _build_where(company: str | None, fiscal_year: str | None) -> dict:
    """Merge the default low-confidence exclusion with the optional FR5 filters."""
    clauses = [DEFAULT_RETRIEVAL_FILTER]
    if company:
        clauses.append({"company": company})
    if fiscal_year:
        clauses.append({"fiscal_year": fiscal_year})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def retrieve(
    question: str,
    cfg: Config,
    collection,
    embedder: Embedder,
    company: str | None = None,
    fiscal_year: str | None = None,
    top_k: int | None = None,
    max_distance: float | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k chunks most relevant to ``question`` (FR4, FR5, FR6).

    Chunks tagged ``text_quality="low_confidence"`` by ingestion (see
    :data:`finsight.ingest.DEFAULT_RETRIEVAL_FILTER`) are excluded by default,
    since we can't guarantee their numbers are as printed in the source.

    Chunks farther than ``max_distance`` (default: ``cfg.max_distance``) are
    dropped too, so an out-of-scope question reliably ends up with no chunks at
    all - and therefore the not-found answer - instead of handing the LLM a
    question with nothing relevant to answer it from and hoping it refuses.
    """
    result = collection.query(
        query_embeddings=[embedder.embed_query(question)],
        n_results=top_k or cfg.top_k,
        where=_build_where(company, fiscal_year),
        include=["documents", "metadatas", "distances"],
    )
    if not result["ids"] or not result["ids"][0]:
        return []
    threshold = cfg.max_distance if max_distance is None else max_distance
    return [
        RetrievedChunk(text=doc, metadata=meta, distance=dist)
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
        if dist <= threshold
    ]
