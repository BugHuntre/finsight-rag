import json

from run_eval import _page, load_golden_set, write_results


def test_load_golden_set_parses_jsonl(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id": "q01", "question": "A?"}\n\n{"id": "q02", "question": "B?"}\n', encoding="utf-8"
    )
    items = load_golden_set(path)
    assert [i["id"] for i in items] == ["q01", "q02"]


def test_page_prefers_printed_page():
    assert _page({"page": 45, "printed_page": 75}) == 75


def test_page_falls_back_to_pdf_page_when_printed_page_unknown():
    assert _page({"page": 45, "printed_page": 0}) == 45


def test_write_results_creates_json_and_appends_csv_row(tmp_path, monkeypatch):
    import run_eval

    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    config_desc = {"top_k": 8}
    result = {"summary": {"hit_rate_at_k": 1.0}, "per_question": []}

    write_results("test_label", config_desc, result)

    data = json.loads((tmp_path / "test_label.json").read_text(encoding="utf-8"))
    assert data["label"] == "test_label"
    assert data["summary"]["hit_rate_at_k"] == 1.0

    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "test_label" in csv_text
    assert "hit_rate_at_k" in csv_text


def test_write_results_appends_without_rewriting_header(tmp_path, monkeypatch):
    import run_eval

    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    result = {"summary": {"hit_rate_at_k": 1.0}, "per_question": []}
    write_results("first", {"top_k": 8}, result)
    write_results("second", {"top_k": 4}, result)

    lines = (tmp_path / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # header + 2 rows
