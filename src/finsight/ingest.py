"""Ingestion pipeline: load PDFs, clean, chunk, embed, store in ChromaDB.

Usage::

    python -m finsight.ingest --path data/raw

PDF files must be named ``<Company>_<FY####>_<annual_report|transcript>.pdf``,
for example ``Infosys_FY2025_annual_report.pdf`` or
``HDFC-Bank_FY2025_transcript.pdf`` (hyphens in the company part become spaces).
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finsight.config import Config, load_config
from finsight.embeddings import Embedder

FILENAME_RE = re.compile(
    r"^(?P<company>.+?)_(?P<year>FY\d{4})_(?P<doc_type>annual_report|transcript)$",
    re.IGNORECASE,
)
DOC_TYPE_LABELS = {"annual_report": "Annual Report", "transcript": "Earnings Call Transcript"}

# Header/footer lines are only looked for at the top and bottom of each page, so
# repeated rows inside financial tables are never stripped.
EDGE_LINES = 3
MIN_PAGES_FOR_HEADER_DETECTION = 5
PAGE_NUMBER_RE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class DocumentInfo:
    """Metadata derived from a PDF file name."""

    company: str
    ticker: str
    fiscal_year: str
    doc_type: str
    source_file: str

    @property
    def document(self) -> str:
        """Human-readable document name, e.g. ``Annual Report FY2025``."""
        return f"{DOC_TYPE_LABELS[self.doc_type]} {self.fiscal_year}"


@dataclass(frozen=True)
class Chunk:
    """A text chunk with its metadata, ready to embed and store."""

    id: str
    text: str
    metadata: dict[str, str | int]


def parse_filename(path: Path, tickers: dict[str, str] | None = None) -> DocumentInfo:
    """Extract company, fiscal year and document type from a PDF file name.

    Raises:
        ValueError: if the file name does not follow the naming convention.
    """
    m = FILENAME_RE.match(path.stem)
    if not m:
        raise ValueError(
            f"{path.name}: expected '<Company>_<FY####>_<annual_report|transcript>.pdf'"
        )
    company = m["company"].replace("-", " ").strip()
    ticker = (tickers or {}).get(company, "")
    return DocumentInfo(
        company=company,
        ticker=ticker,
        fiscal_year=m["year"].upper(),
        doc_type=m["doc_type"].lower(),
        source_file=path.name,
    )


def file_hash(path: Path) -> str:
    """SHA-256 of the file contents, used to skip already-indexed documents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return ``(page_number, raw_text)`` for each page, 1-indexed."""
    with pymupdf.open(path) as doc:
        return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]


def _normalise_edge(line: str) -> str:
    """Key used to match a header/footer across pages (digits masked)."""
    return re.sub(r"\d+", "#", line.strip().lower())


def detect_running_lines(pages: list[str], min_ratio: float) -> set[str]:
    """Find header/footer lines repeated across many pages (normalised keys)."""
    if len(pages) < MIN_PAGES_FOR_HEADER_DETECTION:
        return set()
    counts: Counter[str] = Counter()
    for text in pages:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counts.update({_normalise_edge(ln) for ln in edges})
    threshold = max(3, min_ratio * len(pages))
    return {key for key, n in counts.items() if n >= threshold and key != "#"}


def clean_page(text: str, running_lines: set[str]) -> str:
    """Remove running headers/footers and bare page numbers; normalise whitespace."""
    lines = text.splitlines()
    keep: list[str] = []
    last_edge_start = max(len(lines) - EDGE_LINES, 0)
    for i, line in enumerate(lines):
        at_edge = i < EDGE_LINES or i >= last_edge_start
        if at_edge and line.strip():
            if PAGE_NUMBER_RE.match(line) or _normalise_edge(line) in running_lines:
                continue
        keep.append(line)
    out = "\n".join(keep)
    out = re.sub(r"[ \t ]+", " ", out)
    out = re.sub(r" ?\n ?", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def chunk_document(
    pages: list[tuple[int, str]], info: DocumentInfo, doc_hash: str, cfg: Config
) -> list[Chunk]:
    """Clean pages and split them into chunks carrying full metadata.

    Chunking is done per page so every chunk maps to exactly one page number.
    """
    running = detect_running_lines([t for _, t in pages], cfg.header_footer_min_ratio)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
    chunks: list[Chunk] = []
    for page_no, raw in pages:
        text = clean_page(raw, running)
        if not text:
            continue
        for idx, piece in enumerate(splitter.split_text(text)):
            chunk_id = f"{doc_hash[:8]}-p{page_no:04d}-c{idx:02d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    metadata={
                        "company": info.company,
                        "ticker": info.ticker,
                        "fiscal_year": info.fiscal_year,
                        "doc_type": info.doc_type,
                        "document": info.document,
                        "source_file": info.source_file,
                        "page": page_no,
                        "chunk_id": chunk_id,
                        "file_hash": doc_hash,
                    },
                )
            )
    return chunks


def get_collection(cfg: Config):
    """Open (or create) the persistent Chroma collection."""
    import chromadb

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.index_dir))
    return client.get_or_create_collection(cfg.collection, metadata={"hnsw:space": "cosine"})


def is_indexed(collection, doc_hash: str) -> bool:
    """True if any chunk from a file with this hash is already stored."""
    return bool(collection.get(where={"file_hash": doc_hash}, limit=1)["ids"])


def ingest_pdf(path: Path, cfg: Config, collection, embedder: Embedder) -> int:
    """Ingest a single PDF. Returns the number of chunks added (0 if skipped)."""
    info = parse_filename(path, cfg.tickers)
    doc_hash = file_hash(path)
    if is_indexed(collection, doc_hash):
        print(f"[skip] {path.name}: already indexed")
        return 0

    chunks = chunk_document(load_pdf_pages(path), info, doc_hash, cfg)
    if not chunks:
        print(f"[warn] {path.name}: no extractable text (scanned PDF? needs OCR)")
        return 0

    step = cfg.embed_batch_size
    for start in range(0, len(chunks), step):
        batch = chunks[start : start + step]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=embedder.embed_documents([c.text for c in batch]),
        )
    print(f"[done] {path.name}: {len(chunks)} chunks")
    return len(chunks)


def ingest_folder(
    folder: Path, cfg: Config, collection=None, embedder: Embedder | None = None
) -> dict[str, int]:
    """Ingest every PDF in ``folder``. Returns ``{file name: chunks added}``."""
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {folder}")
    collection = collection if collection is not None else get_collection(cfg)
    embedder = embedder or Embedder(cfg.embedding_model, cfg.embed_batch_size)

    results: dict[str, int] = {}
    for pdf in pdfs:
        try:
            results[pdf.name] = ingest_pdf(pdf, cfg, collection, embedder)
        except ValueError as exc:  # bad file name: report and continue
            print(f"[error] {exc}")
    print(f"Total chunks in index: {collection.count()}")
    return results


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Index PDFs into the FinSight vector store.")
    p.add_argument("--path", type=Path, default=None, help="Folder of PDFs (default: raw_dir)")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--chunk-overlap", type=int, default=None)
    p.add_argument("--collection", type=str, default=None, help="Use a separate collection")
    p.add_argument("--index-dir", type=Path, default=None)
    args = p.parse_args(argv)

    overrides = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "collection": args.collection,
        "index_dir": args.index_dir.resolve() if args.index_dir else None,
    }
    cfg = load_config(args.config, **overrides)
    ingest_folder(args.path or cfg.raw_dir, cfg)


if __name__ == "__main__":
    main()
