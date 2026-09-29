import pytest

from finsight.ingest import get_collection
from finsight.pipeline import answer_question
from tests.conftest import FakeEmbedder, FakeLLM


@pytest.fixture
def collection(cfg):
    coll = get_collection(cfg)
    emb = FakeEmbedder()
    coll.add(
        ids=["c1"],
        documents=["TCS revenue grew to Rs 2,20,938 crore in FY2026."],
        metadatas=[
            {
                "company": "TCS",
                "ticker": "TCS",
                "fiscal_year": "FY2026",
                "doc_type": "annual_report",
                "document": "Annual Report FY2026",
                "source_file": "TCS.pdf",
                "page": 24,
                "printed_page": 45,
                "chunk_id": "c1",
                "file_hash": "abc",
                "content_type": "text",
                "text_quality": "ok",
            }
        ],
        embeddings=emb.embed_documents(["TCS revenue grew to Rs 2,20,938 crore in FY2026."]),
    )
    return coll


def test_answer_question_returns_cited_result(cfg, collection):
    llm = FakeLLM(reply="TCS's revenue was Rs 2,20,938 crore [1].")
    result = answer_question(
        "What was TCS's revenue?",
        cfg,
        collection=collection,
        embedder=FakeEmbedder(),
        llm_call=llm,
    )
    assert result.answer == "TCS's revenue was Rs 2,20,938 crore [1]."
    assert result.sources[0].company == "TCS"
    assert result.sources[0].page == 45  # printed page preferred over PDF page
    assert result.model == cfg.llm_model
    assert result.latency_ms >= 0


def test_answer_question_to_dict_matches_api_shape(cfg, collection):
    result = answer_question(
        "What was TCS's revenue?",
        cfg,
        collection=collection,
        embedder=FakeEmbedder(),
        llm_call=FakeLLM(reply="Revenue grew [1]."),
    )
    d = result.to_dict()
    assert set(d) == {"answer", "sources", "model", "latency_ms"}
    assert d["sources"][0].keys() == {"id", "company", "document", "page", "snippet"}


def test_answer_question_empty_question_raises(cfg, collection):
    with pytest.raises(ValueError):
        answer_question(
            "   ", cfg, collection=collection, embedder=FakeEmbedder(), llm_call=FakeLLM()
        )


def test_answer_question_company_filter_returns_not_found_for_other_company(cfg, collection):
    llm = FakeLLM()
    result = answer_question(
        "What was the revenue?",
        cfg,
        collection=collection,
        embedder=FakeEmbedder(),
        company="Infosys",  # only TCS is seeded
        llm_call=llm,
    )
    assert result.sources == []
    assert llm.calls == []
