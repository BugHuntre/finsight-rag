"""Configuration loading: config.yaml, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Typed view of config.yaml."""

    raw_dir: Path
    index_dir: Path
    chunk_size: int = 800
    chunk_overlap: int = 150
    header_footer_min_ratio: float = 0.4
    embed_batch_size: int = 64
    min_chunk_chars: int = 40
    top_k: int = 8
    collection: str = "finsight"
    max_distance: float = 0.32
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "ollama/qwen2.5:3b"
    tickers: dict[str, str] = field(default_factory=dict)
    # POST /ingest is disabled by default - the spec calls for it to be off in the
    # public demo, since anyone could otherwise trigger ingestion on the deployed API.
    enable_ingest_endpoint: bool = False


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load configuration.

    Precedence (highest first): keyword ``overrides``, ``FINSIGHT_<KEY>``
    environment variables, the YAML file.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")) or {}

    flat: dict[str, Any] = {}
    for section in ("paths", "ingest", "retrieval", "models", "api"):
        flat.update(raw.get(section) or {})
    flat["tickers"] = raw.get("tickers") or {}

    for key, current in list(flat.items()):
        env_val = os.environ.get(f"FINSIGHT_{key.upper()}")
        if env_val is None or isinstance(current, dict):
            continue
        if isinstance(current, bool):
            # bool("false") is True in Python, so this needs its own case rather
            # than falling through to the generic type(current)(env_val) below.
            flat[key] = env_val.strip().lower() in ("1", "true", "yes", "on")
        else:
            flat[key] = type(current)(env_val)
    flat.update({k: v for k, v in overrides.items() if v is not None})

    flat["raw_dir"] = _resolve(flat["raw_dir"])
    flat["index_dir"] = _resolve(flat["index_dir"])
    return Config(**flat)
