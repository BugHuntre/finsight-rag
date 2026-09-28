"""Ingestion pipeline: load PDFs, clean, chunk, embed, store in ChromaDB.

Usage::

    python -m finsight.ingest --path data/raw

PDF files must be named ``<Company>_<FY####>_<annual_report|transcript>.pdf``,
for example ``Infosys_FY2025_annual_report.pdf`` or
``HDFC-Bank_FY2025_transcript.pdf`` (hyphens in the company part become spaces).

Tables are detected separately from prose (see :func:`_page_content`) and chunked
with their header row repeated in every piece (see :func:`chunk_table_markdown`),
so a chunk deep inside a balance sheet still says which column is which year.

Chunks whose text shows signs of a broken PDF font mapping are tagged
``text_quality="low_confidence"`` rather than dropped outright (see
:func:`is_low_confidence`). :data:`DEFAULT_RETRIEVAL_FILTER` excludes them; the
Phase 3 retriever should pass it as (or merge it into) its Chroma ``where`` filter.
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

# Header/footer lines are only looked for near the top and bottom of each page, so
# repeated rows inside financial tables (extracted separately, see PageContent) are
# never stripped. 5 lines is enough to reach a stacked breadcrumb + section heading
# + page number block (seen in some reports) while still leaving normal body
# paragraphs, which run much longer, untouched.
EDGE_LINES = 5
MIN_PAGES_FOR_HEADER_DETECTION = 5
PAGE_NUMBER_RE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)

# Chunks shorter than this (after cleaning) are dropped: cover pages, colour-swatch
# names, standalone captions, etc. add noise without enough content to answer anything.
MIN_CHUNK_CHARS_DEFAULT = 40

# Some PDFs embed a subsetted font with a broken ToUnicode CMap: specific glyph
# codes decode to the wrong Unicode character, so digits/letters come out as
# Greek/Cyrillic/Latin-Extended look-alikes (seen in TCS's FY2026 report, e.g.
# "Đture ǁas aďout ₹3ϰ.ϱ Đrore" for "future was about Rs 38.5 crore") and grouped
# numbers get split by a stray space (e.g. "2, 30, 97 4" for "2,30,974"). Verified
# with two independent extractors (PyMuPDF and pdfplumber/pdfminer) - both decode
# the same glyph differently and both are wrong, so this is unrecoverable from the
# PDF alone (would need OCR). We tag affected chunks instead of silently serving
# possibly-wrong numbers, since the prompt contract is to keep numbers exactly as
# printed in the source.
_GARBLED_GLYPH_RE = re.compile(r"[Ā-ɏͰ-ϿЀ-ӿ]")
# Requires an actual comma (Indian lakh/crore grouping, e.g. "2,30,974") next to the
# stray space, not just any two digits with a space between them - a bare
# "\d \d" would also match ordinary dates like "March 31 2026", which is exactly
# what an earlier, looser version of this pattern did (~90% of its flags were false
# positives from dates in table headers repeated across every chunk). The space is
# also restricted to [ \t], not \s, so it never bridges a newline - several reports
# lay out borderless number columns with one value per line, and \s would treat two
# unrelated stacked numbers (e.g. "1\n1,074") as a single broken number.
_SPLIT_DIGIT_RE = re.compile(r"\d,[ \t]?\d{1,3}[ \t]\d\b|\d[ \t]\d,[ \t]?\d{1,3}\b")

# Phase 3's retriever should pass this as (or merge it into) its Chroma `where`
# filter so low-confidence chunks are excluded from answers by default.
DEFAULT_RETRIEVAL_FILTER = {"text_quality": "ok"}


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
class PageContent:
    """A page's prose text (tables excluded, reading order) and its tables."""

    page_no: int
    text: str
    tables: tuple[str, ...] = ()  # each table as GitHub-flavoured markdown


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


def _bbox_overlap_ratio(block_bbox: tuple[float, ...], table_bbox: tuple[float, ...]) -> float:
    """Fraction of ``block_bbox``'s area that falls inside ``table_bbox``."""
    bx0, by0, bx1, by1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    ix0, iy0 = max(bx0, tx0), max(by0, ty0)
    ix1, iy1 = min(bx1, tx1), min(by1, ty1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max((bx1 - bx0) * (by1 - by0), 1.0)
    return inter / area


def _page_content(page_no: int, page: pymupdf.Page) -> PageContent:
    """Split a page into prose text (reading order) and separately-extracted tables.

    Text blocks that mostly overlap a detected table are dropped from the prose
    stream, so table cells are not duplicated as jumbled loose lines; the table's
    own markdown (with its header row) is kept instead.
    """
    try:
        tables = list(page.find_tables().tables)
    except Exception:
        tables = []  # pragma: no cover - defensive: a malformed page shouldn't abort ingestion

    items: list[tuple[float, str, str]] = []  # (y0, kind, content), merged then sorted
    for x0, y0, x1, y1, text, *_rest in page.get_text("blocks"):
        if not text.strip():
            continue
        if any(_bbox_overlap_ratio((x0, y0, x1, y1), t.bbox) > 0.5 for t in tables):
            continue
        items.append((y0, "text", text))

    table_md: list[str] = []
    for t in tables:
        try:
            md = t.to_markdown()
        except Exception:
            continue  # pragma: no cover - defensive
        if md and md.strip():
            items.append((t.bbox[1], "table", md))
            table_md.append(md)

    items.sort(key=lambda item: item[0])
    text = "\n".join(content for _, kind, content in items if kind == "text")
    ordered_tables = tuple(content for _, kind, content in items if kind == "table")
    return PageContent(page_no=page_no, text=text, tables=ordered_tables or tuple(table_md))


def load_pdf_pages(path: Path) -> list[PageContent]:
    """Return each page's prose text and tables, 1-indexed."""
    with pymupdf.open(path) as doc:
        return [_page_content(i + 1, page) for i, page in enumerate(doc)]


def _normalise_edge(line: str) -> str:
    """Key used to match a header/footer across pages: punctuation/whitespace
    stripped and digits masked, so inconsistent spacing (e.g. a letter-spaced
    heading) or a changing page number don't stop two occurrences from matching.
    """
    s = re.sub(r"[^\w]+", "", line.strip().lower())
    return re.sub(r"\d+", "#", s)


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


def extract_printed_page(text: str) -> int:
    """Find a bare page number near the top/bottom of the raw page text.

    Returns the printed page number (which can differ from the PDF's physical
    page index, e.g. front matter offsets it), or 0 if none is found.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for ln in lines[:EDGE_LINES] + lines[-EDGE_LINES:]:
        if PAGE_NUMBER_RE.match(ln):
            m = re.search(r"\d+", ln)
            if m and 1 <= int(m.group()) <= 9999:
                return int(m.group())
    return 0


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


def is_low_confidence(text: str) -> bool:
    """True if a chunk's numbers/text can't be trusted as printed in the source.

    Two distinct causes end up looking similar and are both worth excluding by
    default: (1) a broken PDF font mapping (see ``_GARBLED_GLYPH_RE``), and
    (2) a table with a merged/multi-row header that ``find_tables()`` read as one
    jumbled cell, spilling several numbers into a single ungridded line (e.g. a
    two-year header row merged with the first data row). Neither is fixable by
    changing how we decode this PDF - the first is a source-file problem, the
    second a table-structure one - so affected chunks are tagged rather than
    silently trusted.
    """
    return bool(_GARBLED_GLYPH_RE.search(text) or _SPLIT_DIGIT_RE.search(text))


def chunk_table_markdown(markdown: str, max_chars: int) -> list[str]:
    """Split a markdown table into pieces that each repeat the header row.

    Without this, a long table's header (which column is FY2026 vs FY2025) lands
    in its own chunk and every other chunk of the table is ambiguous.
    """
    lines = [ln for ln in markdown.splitlines() if ln.strip()]
    if len(lines) < 3:  # header + separator + at least one data row
        return [markdown.strip()] if markdown.strip() else []
    header = "\n".join(lines[:2])
    budget = max(max_chars - len(header) - 1, 200)

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines[2:]:
        if current and current_len + len(line) + 1 > budget:
            pieces.append(header + "\n" + "\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        pieces.append(header + "\n" + "\n".join(current))
    return pieces


def chunk_document(
    pages: list[PageContent], info: DocumentInfo, doc_hash: str, cfg: Config
) -> list[Chunk]:
    """Clean pages and split them into chunks carrying full metadata.

    Chunking is done per page so every chunk maps to exactly one page number.
    Table rows are chunked separately from prose (see :func:`chunk_table_markdown`)
    so column headers stay attached to every piece of a table.
    """
    running = detect_running_lines([p.text for p in pages], cfg.header_footer_min_ratio)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
    min_chars = cfg.min_chunk_chars
    chunks: list[Chunk] = []
    for page in pages:
        base_meta: dict[str, str | int] = {
            "company": info.company,
            "ticker": info.ticker,
            "fiscal_year": info.fiscal_year,
            "doc_type": info.doc_type,
            "document": info.document,
            "source_file": info.source_file,
            "page": page.page_no,
            "printed_page": extract_printed_page(page.text),
            "file_hash": doc_hash,
        }

        cleaned = clean_page(page.text, running)
        for idx, piece in enumerate(splitter.split_text(cleaned)) if cleaned else []:
            if len(piece) < min_chars:
                continue
            chunk_id = f"{doc_hash[:8]}-p{page.page_no:04d}-c{idx:02d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    metadata={
                        **base_meta,
                        "chunk_id": chunk_id,
                        "content_type": "text",
                        "text_quality": "low_confidence" if is_low_confidence(piece) else "ok",
                    },
                )
            )

        for t_idx, table_md in enumerate(page.tables):
            for c_idx, piece in enumerate(chunk_table_markdown(table_md, cfg.chunk_size)):
                if len(piece) < min_chars:
                    continue
                chunk_id = f"{doc_hash[:8]}-p{page.page_no:04d}-b{t_idx:02d}-{c_idx:02d}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=piece,
                        metadata={
                            **base_meta,
                            "chunk_id": chunk_id,
                            "content_type": "table",
                            "text_quality": "low_confidence"
                            if is_low_confidence(piece)
                            else "ok",
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

    n_low = sum(1 for c in chunks if c.metadata["text_quality"] == "low_confidence")
    n_tables = sum(1 for c in chunks if c.metadata["content_type"] == "table")

    step = cfg.embed_batch_size
    for start in range(0, len(chunks), step):
        batch = chunks[start : start + step]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=embedder.embed_documents([c.text for c in batch]),
        )
    print(
        f"[done] {path.name}: {len(chunks)} chunks "
        f"({n_tables} table, {n_low} low_confidence/excluded-by-default)"
    )
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
