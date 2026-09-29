import pytest
from fastapi.testclient import TestClient

from finsight import api
from finsight.config import load_config
from finsight.ingest import get_collection
from tests.conftest import FakeEmbedder, FakeLLM

TCS_META = {
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


@pytest.fixture(autouse=True)
def _reset_state():
    """`api.state` is a module-level dict populated by lifespan; tests set it
    directly instead (see api.py's `llm_call` comment), and TestClient is used
    without `with`, so lifespan never runs and never overwrites what we set here.
    """
    api.state.clear()
    yield
    api.state.clear()


@pytest.fixture
def client(cfg):
    collection = get_collection(cfg)
    emb = FakeEmbedder()
    collection.add(
        ids=["c1"],
        documents=["TCS revenue grew to Rs 2,20,938 crore in FY2026."],
        metadatas=[TCS_META],
        embeddings=emb.embed_documents(["TCS revenue grew to Rs 2,20,938 crore in FY2026."]),
    )
    api.state["cfg"] = cfg
    api.state["collection"] = collection
    api.state["embedder"] = emb
    api.state["llm_call"] = FakeLLM(reply="TCS's revenue grew [1].")
    return TestClient(api.app)


def test_health_reports_indexed_chunk_count(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "indexed_chunks": 1}


def test_documents_lists_indexed_files_with_chunk_count(client):
    resp = client.get("/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert len(docs) == 1
    assert docs[0]["company"] == "TCS"
    assert docs[0]["chunk_count"] == 1


def test_query_returns_cited_answer(client):
    resp = client.post("/query", json={"question": "What was TCS's revenue?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "TCS's revenue grew [1]."
    assert body["sources"][0]["company"] == "TCS"
    assert body["sources"][0]["page"] == 45
    assert set(body) == {"answer", "sources", "model", "latency_ms"}


def test_query_with_optional_filters(client):
    payload = {"question": "revenue?", "company": "TCS", "fiscal_year": "FY2026", "top_k": 3}
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200


def test_query_rejects_empty_question(client):
    resp = client.post("/query", json={"question": "   "})
    assert resp.status_code == 400


def test_query_returns_503_when_index_is_empty(cfg):
    api.state["cfg"] = cfg
    api.state["collection"] = get_collection(cfg)  # freshly created, empty
    api.state["embedder"] = FakeEmbedder()
    api.state["llm_call"] = FakeLLM()
    resp = TestClient(api.app).post("/query", json={"question": "Anything?"})
    assert resp.status_code == 503


def test_query_returns_502_when_llm_call_fails(client):
    def failing_llm(messages, model):
        raise RuntimeError("upstream timeout")

    api.state["llm_call"] = failing_llm
    resp = client.post("/query", json={"question": "What was TCS's revenue?"})
    assert resp.status_code == 502


def test_ingest_disabled_by_default(client):
    resp = client.post("/ingest")
    assert resp.status_code == 403


def test_ingest_enabled_via_config(cfg, tmp_path, make_pdf):
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    make_pdf(cfg.raw_dir / "Infosys_FY2026_annual_report.pdf", ["Revenue grew steadily. " * 10])
    cfg = load_config(
        raw_dir=cfg.raw_dir,
        index_dir=cfg.index_dir,
        collection=cfg.collection,
        enable_ingest_endpoint=True,
    )
    api.state["cfg"] = cfg
    api.state["collection"] = get_collection(cfg)
    api.state["embedder"] = FakeEmbedder()
    resp = TestClient(api.app).post("/ingest")
    assert resp.status_code == 200
    assert resp.json()["results"]["Infosys_FY2026_annual_report.pdf"] > 0
