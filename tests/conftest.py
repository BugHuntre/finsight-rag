"""Shared test fixtures: tiny synthetic PDFs and a fake embedder (no model download)."""

from __future__ import annotations

import re

import pymupdf
import pytest

from finsight.config import load_config

_CONTEXT_BLOCK_RE = re.compile(r"(?m)^\[(\d+)\] ")


class FakeEmbedder:
    """Deterministic 8-dim embeddings; avoids downloading a real model."""

    model_name = "fake"

    def embed_documents(self, texts):
        return [[float((len(t) + i) % 7) + 1.0 for i in range(8)] for t in texts]

    def embed_query(self, question):
        return self.embed_documents([question])[0]


class FakeLLM:
    """Deterministic stand-in for `default_llm_call`; never touches Ollama/an API.

    Records every call it receives and returns a scripted reply (default: cite
    every context block given), so tests can assert on both without any network
    or GPU dependency (per CLAUDE.md: mock the LLM in tests).
    """

    def __init__(self, reply: str | None = None):
        self.reply = reply
        self.calls: list[tuple[list[dict[str, str]], str]] = []

    def __call__(self, messages, model):
        self.calls.append((messages, model))
        if self.reply is not None:
            return self.reply
        block_numbers = _CONTEXT_BLOCK_RE.findall(messages[-1]["content"])
        citations = "".join(f"[{n}]" for n in block_numbers)
        return f"Here is the answer {citations}."


@pytest.fixture
def cfg(tmp_path):
    return load_config(
        raw_dir=tmp_path / "raw",
        index_dir=tmp_path / "index",
        collection="test",
        chunk_size=200,
        chunk_overlap=40,
    )


@pytest.fixture
def make_pdf():
    """Factory: make_pdf(path, pages=[text, ...]) writes a PDF with one text block per page."""

    def _make(path, pages):
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=10)
        doc.save(path)
        doc.close()
        return path

    return _make
