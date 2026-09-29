from finsight.generator import NOT_FOUND_MESSAGE, generate_answer, map_citations
from finsight.retriever import RetrievedChunk
from tests.conftest import FakeLLM


def _chunk(company="TCS", page=10, text="Revenue grew to Rs 100 crore."):
    return RetrievedChunk(
        text=text,
        metadata={
            "company": company,
            "document": "Annual Report FY2026",
            "page": page,
            "printed_page": 0,
        },
        distance=0.2,
    )


def test_map_citations_extracts_only_cited_sources():
    chunks = [_chunk(company="TCS"), _chunk(company="Infosys"), _chunk(company="HDFC Bank")]
    sources = map_citations("TCS grew [1]. Infosys held steady [2].", chunks)
    assert [s.id for s in sources] == [1, 2]
    assert sources[0].company == "TCS"
    assert sources[1].company == "Infosys"


def test_map_citations_ignores_out_of_range_numbers():
    chunks = [_chunk()]
    sources = map_citations("Some hallucinated claim [1][99].", chunks)
    assert [s.id for s in sources] == [1]


def test_map_citations_deduplicates_and_sorts():
    chunks = [_chunk(), _chunk(), _chunk()]
    sources = map_citations("Mentioned again [3][1][1][3].", chunks)
    assert [s.id for s in sources] == [1, 3]


def test_map_citations_truncates_snippet():
    chunks = [_chunk(text="x" * 500)]
    sources = map_citations("[1]", chunks)
    assert len(sources[0].snippet) == 280


def test_map_citations_returns_empty_when_answer_has_no_citations():
    assert map_citations(NOT_FOUND_MESSAGE, [_chunk()]) == []


def test_generate_answer_calls_llm_and_maps_citations():
    llm = FakeLLM(reply="TCS revenue grew [1].")
    chunks = [_chunk(company="TCS")]
    result = generate_answer("What was revenue?", chunks, "fake-model", llm_call=llm)
    assert result.answer == "TCS revenue grew [1]."
    assert result.model == "fake-model"
    assert len(result.sources) == 1
    assert result.sources[0].company == "TCS"
    assert result.latency_ms >= 0
    assert len(llm.calls) == 1


def test_generate_answer_skips_llm_call_when_no_chunks():
    llm = FakeLLM()
    result = generate_answer("Anything?", [], "fake-model", llm_call=llm)
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []
    assert result.latency_ms == 0
    assert llm.calls == []  # no wasted LLM call when there's nothing to ground an answer in
