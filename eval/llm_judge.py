"""RAGAS-equivalent generation-quality metrics, via a direct LiteLLM judge call.

Reimplemented rather than using the `ragas` package: both ragas 0.4.3 (latest)
and 0.2.15 (an older, supposedly more stable release) fail to import in this
environment, because `ragas.llms.base` unconditionally imports
`langchain_community.chat_models.vertexai.ChatVertexAI` at module load time, and
recent langchain-community releases removed that class entirely (it now lives in
the separate `langchain-google-vertexai` package under a different name).
Installing that package doesn't fix the import, since langchain-community's own
namespace is what's missing it - and forcibly downgrading langchain-community
pulls its dependents' version floor down with it and breaks unrelated packages
(litellm's own langchain extras among them). See docs/notes.md for the full
debugging trail. Faithfulness and answer relevancy are each just a judge prompt
plus JSON parsing, simple enough that reimplementing them directly against the
LiteLLM call already proven in generator.py is more reliable here than fighting
that dependency tree.
"""

from __future__ import annotations

import json
import re

from finsight.embeddings import Embedder
from finsight.generator import LLMCallFn, default_llm_call

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_json_array(text: str) -> list:
    """Extract the first JSON array from a judge response (often wrapped in prose)."""
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


FAITHFULNESS_PROMPT = """You are grading whether an AI assistant's answer is faithful to \
the context it was given.

Context:
{context}

Answer to grade:
{answer}

Break the answer into its individual factual claims, then decide for each claim whether it \
is directly supported by the context above. A claim is supported only if the context states \
it or something that clearly implies it - not if it merely sounds plausible.

Respond with ONLY a JSON array, no other text, in this exact form:
[{{"claim": "...", "supported": true}}, {{"claim": "...", "supported": false}}]"""


def faithfulness(
    context: str, answer: str, model: str, llm_call: LLMCallFn = default_llm_call
) -> float | None:
    """Fraction of the answer's claims that the judge finds supported by the context.

    Returns None (not 0.0) if the judge's response couldn't be parsed into any
    claims - that means "couldn't judge this one", which is a different thing
    from "judged and found totally unfaithful" and should not be averaged in as
    if it were.
    """
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response = llm_call([{"role": "user", "content": prompt}], model)
    claims = _parse_json_array(response)
    if not claims:
        return None
    supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported"))
    return supported / len(claims)


RELEVANCY_PROMPT = """Given the answer below, generate {n} different questions that this \
answer would be a good, complete response to.

Answer:
{answer}

Respond with ONLY a JSON array of {n} question strings, no other text."""


def answer_relevancy(
    question: str,
    answer: str,
    model: str,
    embedder: Embedder,
    n: int = 3,
    llm_call: LLMCallFn = default_llm_call,
) -> float | None:
    """Mean cosine similarity between the real question and n "reverse" questions
    the judge generated from the answer (RAGAS's answer-relevancy idea): an
    answer that drifts off-topic produces reverse questions that don't resemble
    what was actually asked.

    Returns None if the judge didn't return any usable questions. Embeddings
    from :class:`Embedder` are L2-normalised, so a plain dot product is the
    cosine similarity.
    """
    prompt = RELEVANCY_PROMPT.format(n=n, answer=answer)
    response = llm_call([{"role": "user", "content": prompt}], model)
    generated = [q for q in _parse_json_array(response) if isinstance(q, str) and q.strip()]
    if not generated:
        return None

    q_vec = embedder.embed_query(question)
    similarities = [
        sum(a * b for a, b in zip(q_vec, embedder.embed_query(gq), strict=True))
        for gq in generated
    ]
    return sum(similarities) / len(similarities)
