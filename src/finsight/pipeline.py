"""End-to-end query pipeline: retrieve -> prompt -> generate -> log (FR4-FR6, FR9).

Usage::

    python -m finsight.pipeline "What was TCS's revenue in FY2026?" --company TCS
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from finsight.config import Config, load_config
from finsight.embeddings import Embedder
from finsight.generator import LLMCallFn, Source, default_llm_call, generate_answer
from finsight.ingest import get_collection
from finsight.retriever import retrieve

logger = logging.getLogger("finsight.pipeline")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs" / "queries.jsonl"


@dataclass(frozen=True)
class QueryResult:
    """Matches the POST /query response shape from the API spec (section 9)."""

    answer: str
    sources: list[Source]
    model: str
    latency_ms: int

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [asdict(s) for s in self.sources],
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


def setup_logging(path: Path = LOG_PATH) -> None:
    """Attach a JSON-lines file handler to the pipeline logger (FR9), once."""
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def answer_question(
    question: str,
    cfg: Config,
    collection=None,
    embedder: Embedder | None = None,
    company: str | None = None,
    fiscal_year: str | None = None,
    top_k: int | None = None,
    max_distance: float | None = None,
    llm_call: LLMCallFn = default_llm_call,
) -> QueryResult:
    """Run the full query pipeline for one question (FR4, FR5, FR6, FR9).

    Raises:
        ValueError: if ``question`` is empty (NFR: meaningful error messages).
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    collection = collection if collection is not None else get_collection(cfg)
    embedder = embedder or Embedder(cfg.embedding_model, cfg.embed_batch_size)

    t0 = time.time()
    chunks = retrieve(
        question, cfg, collection, embedder, company, fiscal_year, top_k, max_distance
    )
    result = generate_answer(question, chunks, cfg.llm_model, llm_call=llm_call)
    total_latency_ms = int((time.time() - t0) * 1000)

    logger.info(
        json.dumps(
            {
                "question": question,
                "company": company,
                "fiscal_year": fiscal_year,
                "retrieved_chunk_ids": [c.metadata["chunk_id"] for c in chunks],
                "model": result.model,
                "latency_ms": total_latency_ms,
            }
        )
    )
    return QueryResult(
        answer=result.answer,
        sources=result.sources,
        model=result.model,
        latency_ms=total_latency_ms,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ask one question and print the cited answer."""
    p = argparse.ArgumentParser(description="Ask FinSight a question from the command line.")
    p.add_argument("question")
    p.add_argument("--company", default=None)
    p.add_argument("--fiscal-year", default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    result = answer_question(
        args.question,
        cfg,
        company=args.company,
        fiscal_year=args.fiscal_year,
        top_k=args.top_k,
        max_distance=args.max_distance,
    )

    print(result.answer)
    if result.sources:
        print("\nSources:")
        for s in result.sources:
            print(f"  [{s.id}] {s.company} - {s.document}, page {s.page}")
    print(f"\n({result.model}, {result.latency_ms} ms)")


if __name__ == "__main__":
    main()
