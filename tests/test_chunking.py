from pathlib import Path

import pytest

from finsight.ingest import (
    DocumentInfo,
    chunk_document,
    clean_page,
    detect_running_lines,
    parse_filename,
)


def _info():
    return DocumentInfo("TCS", "TCS", "FY2025", "annual_report", "TCS_FY2025_annual_report.pdf")


def test_parse_filename_extracts_metadata():
    info = parse_filename(Path("HDFC-Bank_FY2025_transcript.pdf"), {"HDFC Bank": "HDFCBANK"})
    assert info.company == "HDFC Bank"
    assert info.ticker == "HDFCBANK"
    assert info.fiscal_year == "FY2025"
    assert info.doc_type == "transcript"
    assert info.document == "Earnings Call Transcript FY2025"


def test_parse_filename_unknown_ticker_is_empty():
    assert parse_filename(Path("Foo_FY2024_annual_report.pdf")).ticker == ""


@pytest.mark.parametrize(
    "name", ["report.pdf", "TCS_2025_annual_report.pdf", "TCS_FY2025_misc.pdf"]
)
def test_parse_filename_rejects_bad_names(name):
    with pytest.raises(ValueError):
        parse_filename(Path(name))


def test_clean_page_removes_page_numbers_and_collapses_whitespace():
    raw = "Revenue   grew\t12%\n\n\n\nto Rs 100 crore\n42"
    assert clean_page(raw, set()) == "Revenue grew 12%\n\nto Rs 100 crore"


def test_running_headers_removed_but_table_rows_kept():
    def page(i):
        lines = [f"Line {j} of narrative for page {i}." for j in range(10)]
        lines.insert(5, f"Total 1,{i}00")
        return "Acme Annual Report 2025\n" + "\n".join(lines) + f"\nPage {i}"

    pages = [page(i) for i in range(10)]
    running = detect_running_lines(pages, min_ratio=0.4)
    cleaned = clean_page(pages[3], running)
    assert "Acme Annual Report" not in cleaned
    assert "Page 3" not in cleaned
    assert "Line 5 of narrative for page 3." in cleaned
    assert "Total 1,300" in cleaned  # interior lines are never treated as headers/footers


def test_header_detection_skipped_for_short_documents():
    assert detect_running_lines(["Header\nx"] * 3, 0.4) == set()


def test_chunks_carry_full_metadata_and_page_numbers(cfg):
    pages = [(1, "A" * 50), (2, ""), (3, "Sentence about margins. " * 40)]
    chunks = chunk_document(pages, _info(), "abcdef1234567890", cfg)
    assert {c.metadata["page"] for c in chunks} == {1, 3}  # blank page 2 skipped
    assert len(chunks) > 2  # page 3 split into several chunks
    for c in chunks:
        assert len(c.text) <= cfg.chunk_size
        assert c.id == c.metadata["chunk_id"]
        assert c.metadata["company"] == "TCS"
        assert c.metadata["fiscal_year"] == "FY2025"
        assert c.metadata["doc_type"] == "annual_report"
        assert c.metadata["source_file"] == "TCS_FY2025_annual_report.pdf"
    assert len({c.id for c in chunks}) == len(chunks)
