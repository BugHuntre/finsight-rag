"""Grounded prompt construction: system rules + numbered context chunks + question.

Implements the prompt requirements from the requirements doc, section 6.3:
answer only from context, cite inline by chunk number, keep numbers exactly as
written (with units), and stay concise unless detail is asked for.
"""

from __future__ import annotations

from finsight.retriever import RetrievedChunk

NOT_FOUND_MESSAGE = "I could not find this information in the documents."

SYSTEM_PROMPT = f"""You are FinSight, a financial-research assistant that answers questions \
about Indian listed companies using only the numbered context excerpts provided below, taken \
from their annual reports and earnings-call transcripts.

Rules:
- Answer only using the provided context. Do not use outside knowledge or guess. If the answer \
is not in the context, reply with exactly this sentence and nothing else: "{NOT_FOUND_MESSAGE}"
- Cite every factual claim inline using the context number(s) it came from, e.g. [1] or [2][3].
- Keep numbers exactly as written in the source, including their units (crore, lakh, %, Rs, \
US$, etc) - never convert, round, or recompute them.
- Be concise: 2-6 sentences, unless the question explicitly asks for a detailed breakdown."""


def page_label(metadata: dict) -> int:
    """Prefer the printed page number when we found one; fall back to the PDF page."""
    return metadata.get("printed_page") or metadata["page"]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered context blocks, 1-indexed."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        m = chunk.metadata
        header = f"[{i}] {m['company']} - {m['document']} (page {page_label(m)})"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    """Build the chat messages to send to the LLM."""
    if chunks:
        user = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    else:
        user = f"No relevant context was found for this question.\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
