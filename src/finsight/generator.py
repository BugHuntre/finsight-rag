"""LLM calls and citation mapping (FR4, FR6).

Model selection goes through LiteLLM so ``models.llm_model`` in config.yaml can
name any provider LiteLLM supports without a code change - "ollama/qwen2.5:3b"
for local dev, "groq/llama-3.1-8b-instant" for the deployed demo (free hosting
can't run Ollama). See CLAUDE.md for the hardware constraint behind that choice.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from finsight.prompts import NOT_FOUND_MESSAGE, build_messages, page_label
from finsight.retriever import RetrievedChunk

CITATION_RE = re.compile(r"\[(\d+)\]")
OLLAMA_API_BASE = "http://localhost:11434"
SNIPPET_CHARS = 280

# Injectable so tests never need a real Ollama server or API key (see
# tests/conftest.py's FakeLLM). Signature: (messages, model_name) -> answer text.
LLMCallFn = Callable[[list[dict[str, str]], str], str]


@dataclass(frozen=True)
class Source:
    """One cited source, matching the API spec's `sources[]` shape (section 9)."""

    id: int
    company: str
    document: str
    page: int
    snippet: str


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    sources: list[Source]
    model: str
    latency_ms: int


def default_llm_call(messages: list[dict[str, str]], model: str, retries: int = 1) -> str:
    """Call the configured LLM via LiteLLM, with one retry on a transient error (NFR)."""
    import litellm

    kwargs = {"api_base": OLLAMA_API_BASE} if model.startswith("ollama/") else {}
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = litellm.completion(model=model, messages=messages, timeout=90, **kwargs)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001 - retry any transient provider error
            last_exc = exc
            if attempt == retries:
                raise
    raise last_exc  # pragma: no cover - unreachable, satisfies type checkers


def map_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Source]:
    """Build the sources list for the citation numbers the answer actually used.

    Out-of-range numbers (a hallucinated [99]) are silently dropped rather than
    raising, since the answer text itself is still shown to the user either way.
    """
    cited = sorted({int(n) for n in CITATION_RE.findall(answer)} & set(range(1, len(chunks) + 1)))
    sources = []
    for n in cited:
        chunk = chunks[n - 1]
        m = chunk.metadata
        sources.append(
            Source(
                id=n,
                company=m["company"],
                document=m["document"],
                page=page_label(m),
                snippet=chunk.text[:SNIPPET_CHARS],
            )
        )
    return sources


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
    llm_call: LLMCallFn = default_llm_call,
) -> GenerationResult:
    """Build the grounded prompt, call the LLM, and map citations to sources (FR4, FR6).

    Returns the "not found" answer without calling the LLM at all when nothing was
    retrieved - there is nothing it could ground an answer in anyway.
    """
    if not chunks:
        return GenerationResult(answer=NOT_FOUND_MESSAGE, sources=[], model=model, latency_ms=0)

    messages = build_messages(question, chunks)
    t0 = time.time()
    answer = llm_call(messages, model)
    latency_ms = int((time.time() - t0) * 1000)
    return GenerationResult(
        answer=answer, sources=map_citations(answer, chunks), model=model, latency_ms=latency_ms
    )
