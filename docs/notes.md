# Interview notes

Fill in short answers as you build (spec section 16). Write down what you *measured*, not what you expect.

## Why this chunk size and overlap? What happened when I changed them?

Starting point: 800 chars / 150 overlap (spec default). Chunking is done per page so every chunk
maps to exactly one page number, which keeps citations exact.

E2 results (hit rate@8 / MRR, blind retrieval, no company filter, 45 in-scope golden questions):
chunk 400 -> 0.56 / 0.41; chunk 800 (baseline) -> 0.58 / 0.47; chunk 1200 -> **0.62** / 0.42.

1200 wins on hit rate but not MRR. My read: a bigger chunk is more likely to *contain* the right
page somewhere in the top-8 (more text per chunk, more chances to overlap with the golden page),
but it isn't a more precise semantic match once retrieved, so it doesn't rank any higher - and
sometimes lower, since a larger, more topically-diffuse chunk dilutes the embedding. 400 was worse
on both: small chunks fragment context enough that individual chunks become less distinctive.

I kept 800 for now rather than switching on this alone (see README's "Configuration chosen"
paragraph) - the live index is already built at 800, and this is one signal from an unaudited
golden set, not a strong enough basis by itself.

## How do I stop the model hallucinating, and how did I measure it?

Two layers, because the prompt rule alone was not enough:

1. A grounded system prompt (`prompts.py`): answer only from the numbered context given,
   cite every claim inline as `[n]`, and reply with an exact refusal sentence if the answer
   isn't there.
2. A retrieval-side similarity cutoff (`retrieval.max_distance` in `config.yaml`), checked
   *before* the LLM is ever called.

Layer 2 exists because layer 1 failed in testing. Asked "What is the capital of France?"
against the real 5-report index with no cutoff, qwen2.5:3b answered "The capital of France
is **E R**" and cited two irrelevant chunks - it did not refuse, despite the explicit
instruction. I measured retrieval distances for 4 relevant finance questions (0.20-0.26)
against 4 unrelated questions (0.38-0.53) on the real index and found a clean gap, so
chunks beyond 0.32 cosine distance are dropped before generation; if that empties the
retrieved set, the pipeline returns the not-found answer without calling the LLM at all
(cheaper too: ~6s instead of ~25s, since no generation happens).

This was a hand-picked default from 8 examples at the time; the golden set (Phase 5) now backs it
up more directly - refusal accuracy on its 5 out-of-scope questions is 1.00 (5/5).

But the cutoff only stops the *refuse-when-nothing-relevant* failure. It doesn't stop a more subtle
one I only found by reading the golden-set answers, not from a metric: with company filtering off
(the eval's default - see README), 2 of 10 comparison questions got a confidently-cited but
factually *wrong* answer because retrieval let a different company's chunk into the context and the
small model attributed a real number to the wrong company (asked about TCS's profit, it answered
with Infosys's real figures, ₹26,713cr -> ₹29,440cr, mislabeled as TCS's). The faithfulness judge
only scored this 0.5, not 0 - it checks whether a claim is *supported by the given context*, not
whether that context is about the right company, so a wrong-company answer built from real numbers
partially passes. That's a real gap in what RAGAS-style faithfulness catches, not just a model
problem. The practical fix isn't a better cutoff - it's making the company filter (FR5) required or
auto-detected from the question, rather than optional, specifically for this failure mode.

## Why a reranker? Cross-encoder vs embedding model?

Not measured - E4 wasn't run (stretch goal, see README). Conceptually, though: a bi-encoder
(bge-small, what I use for retrieval) embeds the question and each chunk *independently*, so
similarity is a single dot product with no interaction between them - that's what makes it cheap
enough to run against 14k+ chunks. A cross-encoder (bge-reranker-base) instead takes the question
and one candidate chunk *together* as one input and outputs a single relevance score with full
attention between them, which is far more accurate but far too slow to run over the whole index -
hence rerank only the top-k a bi-encoder already narrowed down, not the full corpus.

## Why did hybrid search help (or not) with numbers and company names?

Not measured - E5 wasn't run either. But the TCS/Infosys mixup above (see the hallucination note)
is a concrete, real motivating case for exactly this: BM25 matches exact tokens like "TCS" or
"2,20,938", which a semantic embedding can blur across similar-topic chunks from other companies.
A hybrid retriever wouldn't fix the model's attribution error by itself, but it should make it less
likely that a same-topic-different-company chunk crowds out the right one in the first place. That's
a hypothesis motivated by a real failure, not a measurement - it belongs in the same follow-up as E4.

## How would I scale to 10,000 documents?

Measured baseline first: ingesting the 5 real reports (~1,900 pages, 14,223 chunks after cleaning)
took about 34 minutes total, almost entirely embedding time (~12-15 chunks/sec on this machine's
CPU) - table detection via `find_tables()` roughly doubled that. At 10,000 documents (assume ~300
pages each = 3M pages), that's naive linear scaling to weeks of single-machine, single-threaded
ingestion, which is the real bottleneck, not query-time retrieval (Chroma's HNSW index barely
notices going from 14k to a few million vectors). Concretely: batch ingestion across worker
processes (embedding batches are already batched per-file; parallelize across files, which
`ingest_folder`'s per-PDF loop makes easy since each PDF's `file_hash` check is independent), move
off single-node ChromaDB to a managed/clustered vector DB (e.g. Qdrant, pgvector, or Chroma's own
hosted offering) so retrieval and embedding storage aren't sharing one process's memory, and cache
embeddings for repeat/near-duplicate content (multiple companies' boilerplate disclosures repeat
near-verbatim). Query-time answer generation would also need request queuing/async batching if
served at any real concurrency, since a single Ollama/API call already takes seconds.

## How would I handle tables in PDFs?

Implemented, not hypothetical (see `ingest.py`'s `_page_content`/`chunk_table_markdown`, and the
"Ingestion quality fixes" commit). PyMuPDF's `find_tables()` detects each table's bounding box and
structure separately from surrounding prose, which I convert to markdown and chunk with the header
row repeated in every piece - so a chunk deep inside a 49-row balance sheet still says which column
is FY2026 vs FY2025. This isn't free: it roughly doubled ingestion time, and it isn't perfect -
`find_tables()` sometimes merges a multi-row header into one jumbled cell (a different reports'
tables spilled several numbers into an ungridded line), which I couldn't fix by changing extraction
settings, so I tag those chunks `text_quality="low_confidence"` and exclude them from retrieval by
default rather than risk citing a wrong number with false confidence.

## What would I change for production?

In priority order, based on what actually broke or fell short of target during this build, not a
generic list: (1) make the company filter required-or-auto-detected rather than optional, given the
cross-company hallucination this evaluation caught (the single highest-value fix relative to
effort); (2) re-run faithfulness/answer relevancy against a real hosted judge (Groq/Gemini) instead
of qwen2.5:3b grading itself - the spec called for this and I couldn't do it here for lack of an API
key, and self-grading is a known weak measurement; (3) run E4/E5 (reranking, hybrid search) before
committing to any chunk-size change, since E2's chunk-1200 result traded MRR for hit rate and a
reranker might make that trade-off moot; (4) OCR fallback for the font-corruption case found in
TCS's PDF (a handful of pages decode wrong at the *source* level, not fixable by extraction
settings - see the ingestion-fixes commit); (5) the 10,000-document scaling changes above, once
there's an actual second client/company pushing past what one machine handles comfortably.
