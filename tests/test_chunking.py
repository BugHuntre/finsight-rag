from pathlib import Path

import pytest

from finsight.ingest import (
    DocumentInfo,
    PageContent,
    chunk_document,
    chunk_table_markdown,
    clean_page,
    detect_running_lines,
    extract_printed_page,
    is_low_confidence,
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
    pages = [
        PageContent(1, "A" * 50),
        PageContent(2, ""),
        PageContent(3, "Sentence about margins. " * 40),
    ]
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
        assert c.metadata["content_type"] == "text"
        assert c.metadata["text_quality"] == "ok"
    assert len({c.id for c in chunks}) == len(chunks)


def test_short_chunks_are_dropped(cfg):
    pages = [PageContent(1, "Too short."), PageContent(2, "Sentence about margins. " * 40)]
    chunks = chunk_document(pages, _info(), "abcdef1234567890", cfg)
    assert all(c.metadata["page"] != 1 for c in chunks)
    assert min(len(c.text) for c in chunks) >= cfg.min_chunk_chars


def test_low_confidence_text_is_tagged_not_dropped(cfg):
    garbled = "Đture ǁas aďout ₹3ϰ.ϱ Đrore, up from the previous year's figure." * 3
    pages = [PageContent(1, garbled)]
    chunks = chunk_document(pages, _info(), "abcdef1234567890", cfg)
    assert chunks  # kept, just flagged
    assert all(c.metadata["text_quality"] == "low_confidence" for c in chunks)


def test_table_chunks_repeat_header_and_are_flagged_as_tables(cfg):
    header = "|Particulars|FY2026|FY2025|\n|---|---|---|"
    rows = "\n".join(f"|Line item {i}|{i * 10}|{i * 9}|" for i in range(60))
    pages = [PageContent(1, "Balance sheet follows.", tables=(header + "\n" + rows,))]
    chunks = chunk_document(pages, _info(), "abcdef1234567890", cfg)
    table_chunks = [c for c in chunks if c.metadata["content_type"] == "table"]
    assert len(table_chunks) > 1  # the table is long enough to need more than one piece
    for c in table_chunks:
        assert c.text.startswith("|Particulars|FY2026|FY2025|")
        assert len(c.text) <= cfg.chunk_size + len(header) + 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Revenue grew to Rs 1,62,990 crore.", False),
        ("Đture ǁas aďout ₹3ϰ.ϱ Đrore.", True),  # broken font glyphs
        ("Profit before tax was 2, 30, 97 4 crore.", True),  # digit group split by a stray space
    ],
)
def test_is_low_confidence(text, expected):
    assert is_low_confidence(text) is expected


def test_extract_printed_page_finds_bare_number_near_edges():
    text = "Statutory Reports\nCorporate Overview\nFinancial Statements\nBoard's Report\n75\nBody."
    assert extract_printed_page(text) == 75


def test_extract_printed_page_returns_zero_when_absent():
    assert extract_printed_page("Just body text, no page number.") == 0


def test_chunk_table_markdown_repeats_header_in_every_piece():
    header = "|A|B|\n|---|---|"
    rows = "\n".join(f"|row{i}|{i}|" for i in range(50))
    pieces = chunk_table_markdown(header + "\n" + rows, max_chars=200)
    assert len(pieces) > 1
    assert all(p.startswith(header) for p in pieces)
    assert all(len(p) <= 200 + len(header) + 1 for p in pieces)


def test_chunk_table_markdown_small_table_is_one_piece():
    small = "|A|B|\n|---|---|\n|1|2|"
    assert chunk_table_markdown(small, max_chars=800) == [small]
