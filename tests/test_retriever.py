import pytest

from finsight.config import load_config
from finsight.ingest import get_collection
from finsight.retriever import retrieve
from tests.conftest import FakeEmbedder


@pytest.fixture
def collection(cfg):
    return get_collection(cfg)


def _seed(collection, embedder, rows):
    """rows: list of (id, text, metadata) tuples."""
    ids, texts, metas = zip(*rows, strict=True)
    collection.add(
        ids=list(ids),
        documents=list(texts),
        metadatas=list(metas),
        embeddings=embedder.embed_documents(list(texts)),
    )


def _meta(company="TCS", fiscal_year="FY2026", page=1, text_quality="ok"):
    return {
        "company": company,
        "ticker": company[:3].upper(),
        "fiscal_year": fiscal_year,
        "doc_type": "annual_report",
        "document": f"Annual Report {fiscal_year}",
        "source_file": f"{company}.pdf",
        "page": page,
        "printed_page": 0,
        "chunk_id": f"{company}-{page}",
        "file_hash": "abc123",
        "content_type": "text",
        "text_quality": text_quality,
    }


def test_retrieve_returns_top_k(cfg, collection):
    emb = FakeEmbedder()
    rows = [(f"c{i}", f"Revenue text number {i}", _meta(page=i)) for i in range(5)]
    _seed(collection, emb, rows)
    results = retrieve("revenue", cfg, collection, emb, top_k=3)
    assert len(results) == 3
    assert all(r.metadata["company"] == "TCS" for r in results)


def test_retrieve_filters_by_company(cfg, collection):
    emb = FakeEmbedder()
    rows = [
        ("c1", "TCS revenue grew", _meta(company="TCS")),
        ("c2", "Infosys revenue grew", _meta(company="Infosys")),
    ]
    _seed(collection, emb, rows)
    results = retrieve("revenue", cfg, collection, emb, company="Infosys", top_k=5)
    assert len(results) == 1
    assert results[0].metadata["company"] == "Infosys"


def test_retrieve_filters_by_fiscal_year(cfg, collection):
    emb = FakeEmbedder()
    rows = [
        ("c1", "old year figures", _meta(fiscal_year="FY2025")),
        ("c2", "new year figures", _meta(fiscal_year="FY2026")),
    ]
    _seed(collection, emb, rows)
    results = retrieve("figures", cfg, collection, emb, fiscal_year="FY2025", top_k=5)
    assert len(results) == 1
    assert results[0].metadata["fiscal_year"] == "FY2025"


def test_retrieve_excludes_low_confidence_by_default(cfg, collection):
    emb = FakeEmbedder()
    rows = [
        ("c1", "clean chunk", _meta(text_quality="ok")),
        ("c2", "garbled chunk", _meta(text_quality="low_confidence")),
    ]
    _seed(collection, emb, rows)
    results = retrieve("chunk", cfg, collection, emb, top_k=5)
    assert len(results) == 1
    assert results[0].metadata["text_quality"] == "ok"


def test_retrieve_returns_empty_list_when_no_match(cfg, collection):
    emb = FakeEmbedder()
    _seed(collection, emb, [("c1", "some text", _meta(company="TCS"))])
    assert retrieve("anything", cfg, collection, emb, company="Nonexistent") == []


def test_retrieve_on_empty_collection(cfg, collection):
    emb = FakeEmbedder()
    assert retrieve("anything", cfg, collection, emb) == []


def test_retrieve_drops_chunks_beyond_max_distance(cfg, collection):
    emb = FakeEmbedder()
    _seed(collection, emb, [("c1", "revenue text", _meta())])
    # FakeEmbedder is deterministic but not semantically meaningful, so distances
    # aren't representative here - this only checks the cutoff plumbing works.
    assert retrieve("revenue text", cfg, collection, emb, max_distance=-1) == []
    assert len(retrieve("revenue text", cfg, collection, emb, max_distance=999)) == 1


def test_retrieve_uses_configured_max_distance_by_default(cfg, collection):
    emb = FakeEmbedder()
    _seed(collection, emb, [("c1", "revenue text", _meta())])
    cfg_strict = load_config(
        raw_dir=cfg.raw_dir, index_dir=cfg.index_dir, collection=cfg.collection, max_distance=-1
    )
    assert retrieve("revenue text", cfg_strict, collection, emb) == []
