import chromadb
import pytest

from finsight.ingest import get_collection, ingest_folder, ingest_pdf, load_pdf_pages
from tests.conftest import FakeEmbedder

PAGES = [f"Page {i} discusses operating margin of {i + 20}% for the year. " * 6 for i in range(3)]


@pytest.fixture
def collection(cfg):
    return get_collection(cfg)


def test_load_pdf_pages_is_one_indexed(tmp_path, make_pdf):
    pdf = make_pdf(tmp_path / "TCS_FY2025_annual_report.pdf", PAGES)
    pages = load_pdf_pages(pdf)
    assert [n for n, _ in pages] == [1, 2, 3]
    assert "operating margin" in pages[0][1]


def test_ingest_stores_chunks_with_metadata(cfg, collection, tmp_path, make_pdf):
    pdf = make_pdf(tmp_path / "TCS_FY2025_annual_report.pdf", PAGES)
    added = ingest_pdf(pdf, cfg, collection, FakeEmbedder())
    assert added > 0
    assert collection.count() == added
    meta = collection.get(limit=1)["metadatas"][0]
    assert meta["company"] == "TCS"
    assert meta["ticker"] == "TCS"
    assert meta["page"] in (1, 2, 3)


def test_reingest_is_skipped_by_file_hash(cfg, collection, tmp_path, make_pdf):
    pdf = make_pdf(tmp_path / "TCS_FY2025_annual_report.pdf", PAGES)
    embedder = FakeEmbedder()
    first = ingest_pdf(pdf, cfg, collection, embedder)
    assert ingest_pdf(pdf, cfg, collection, embedder) == 0
    assert collection.count() == first


def test_ingest_folder_reports_bad_names_and_continues(cfg, collection, make_pdf, capsys):
    cfg.raw_dir.mkdir(parents=True)
    make_pdf(cfg.raw_dir / "TCS_FY2025_annual_report.pdf", PAGES)
    make_pdf(cfg.raw_dir / "oops.pdf", PAGES)
    results = ingest_folder(cfg.raw_dir, cfg, collection, FakeEmbedder())
    assert results["TCS_FY2025_annual_report.pdf"] > 0
    assert "oops.pdf" not in results
    assert "[error]" in capsys.readouterr().out


def test_ingest_folder_without_pdfs_raises(cfg, collection):
    cfg.raw_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        ingest_folder(cfg.raw_dir, cfg, collection, FakeEmbedder())


def test_collection_uses_cosine_space(collection):
    assert isinstance(collection, chromadb.Collection)
    assert collection.metadata["hnsw:space"] == "cosine"
