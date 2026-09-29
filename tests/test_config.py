import pytest

from finsight.config import load_config


def test_load_config_defaults(tmp_path):
    cfg = load_config(raw_dir=tmp_path / "raw", index_dir=tmp_path / "index")
    assert cfg.chunk_size == 800
    assert cfg.enable_ingest_endpoint is False


def test_kwarg_overrides_take_precedence(tmp_path):
    cfg = load_config(raw_dir=tmp_path / "raw", index_dir=tmp_path / "index", chunk_size=123)
    assert cfg.chunk_size == 123


@pytest.mark.parametrize(
    "env_val,expected",
    [("true", True), ("True", True), ("1", True), ("yes", True), ("false", False), ("0", False)],
)
def test_bool_env_override_parses_common_spellings(tmp_path, monkeypatch, env_val, expected):
    monkeypatch.setenv("FINSIGHT_ENABLE_INGEST_ENDPOINT", env_val)
    cfg = load_config(raw_dir=tmp_path / "raw", index_dir=tmp_path / "index")
    assert cfg.enable_ingest_endpoint is expected


def test_bool_env_override_of_false_is_not_truthy_string_bug(tmp_path, monkeypatch):
    # Regression test: `bool("false")` is True in Python, so a naive
    # `type(current)(env_val)` cast would incorrectly enable this flag.
    monkeypatch.setenv("FINSIGHT_ENABLE_INGEST_ENDPOINT", "false")
    cfg = load_config(raw_dir=tmp_path / "raw", index_dir=tmp_path / "index")
    assert cfg.enable_ingest_endpoint is False


def test_numeric_env_override_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("FINSIGHT_TOP_K", "3")
    cfg = load_config(raw_dir=tmp_path / "raw", index_dir=tmp_path / "index")
    assert cfg.top_k == 3
