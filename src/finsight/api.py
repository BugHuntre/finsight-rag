"""FastAPI backend (FR1-FR9, endpoints per requirements doc section 9).

Run with::

    uvicorn finsight.api:app --reload

Swagger docs are auto-generated at /docs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from finsight.config import Config, load_config
from finsight.embeddings import Embedder
from finsight.generator import default_llm_call
from finsight.ingest import get_collection, ingest_folder
from finsight.pipeline import answer_question, setup_logging

# Populated at startup by `lifespan`; a plain dict (not app.state) so tests can
# swap in fakes without spinning up a real embedding model or LLM.
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load config, the embedding model, and the Chroma collection once at startup."""
    setup_logging()
    cfg = load_config()
    state["cfg"] = cfg
    state["collection"] = get_collection(cfg)
    state["embedder"] = Embedder(cfg.embedding_model, cfg.embed_batch_size)
    yield
    state.clear()


app = FastAPI(
    title="FinSight RAG API",
    description="Question-answering over Indian company annual reports and earnings-call "
    "transcripts, with page-level citations.",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str
    company: str | None = None
    fiscal_year: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class SourceResponse(BaseModel):
    id: int
    company: str
    document: str
    page: int
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    model: str
    latency_ms: int


class DocumentInfo(BaseModel):
    company: str
    fiscal_year: str
    doc_type: str
    document: str
    source_file: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


class IngestResponse(BaseModel):
    results: dict[str, int]


def _cfg() -> Config:
    return state["cfg"]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Returns status and the number of indexed chunks."""
    return HealthResponse(status="ok", indexed_chunks=state["collection"].count())


@app.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    """Lists indexed documents with company, year, and chunk count."""
    metadatas = state["collection"].get(include=["metadatas"])["metadatas"]
    grouped: dict[tuple[str, str, str, str, str], int] = {}
    for m in metadatas:
        key = (m["company"], m["fiscal_year"], m["doc_type"], m["document"], m["source_file"])
        grouped[key] = grouped.get(key, 0) + 1
    return [
        DocumentInfo(
            company=company,
            fiscal_year=fiscal_year,
            doc_type=doc_type,
            document=document,
            source_file=source_file,
            chunk_count=count,
        )
        for (company, fiscal_year, doc_type, document, source_file), count in sorted(
            grouped.items()
        )
    ]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answers a question with citations (FR4-FR6).

    Returns 400 for an empty question, 503 if nothing has been ingested yet, and
    502 if the LLM call fails after its retry (NFR: meaningful error messages).
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if state["collection"].count() == 0:
        raise HTTPException(status_code=503, detail="index is empty - run ingestion first")
    try:
        result = answer_question(
            req.question,
            _cfg(),
            collection=state["collection"],
            embedder=state["embedder"],
            company=req.company,
            fiscal_year=req.fiscal_year,
            top_k=req.top_k,
            # Not set in production (falls back to the real LLM); tests inject a
            # fake here since answer_question's own default is bound at import
            # time and can't be monkeypatched after the fact.
            llm_call=state.get("llm_call", default_llm_call),
        )
    except Exception as exc:  # noqa: BLE001 - surface as a clean API error, not a 500 traceback
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc
    return QueryResponse(**result.to_dict())


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    """Triggers ingestion of files in data/raw. Disabled by default (off in the public demo)."""
    cfg = _cfg()
    if not cfg.enable_ingest_endpoint:
        raise HTTPException(status_code=403, detail="the /ingest endpoint is disabled")
    results = ingest_folder(cfg.raw_dir, cfg, state["collection"], state["embedder"])
    return IngestResponse(results=results)
