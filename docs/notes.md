# Interview notes

Fill in short answers as you build (spec section 16). Write down what you *measured*, not what you expect.

## Why this chunk size and overlap? What happened when I changed them?

Starting point: 800 chars / 150 overlap (spec default). Chunking is done per page so every chunk
maps to exactly one page number, which keeps citations exact. Results of E2 (400 / 800 / 1200): TBD.

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

This is a hand-picked default from 8 examples, not the golden set - RAGAS faithfulness and
refusal accuracy (G4, G5) in Phase 5 are the real measurement of whether it's calibrated
right, and it should be re-tuned against those results, not left as-is.

## Why a reranker? Cross-encoder vs embedding model?

TBD.

## Why did hybrid search help (or not) with numbers and company names?

TBD.

## How would I scale to 10,000 documents?

TBD (managed vector DB, batched/async ingestion, caching).

## How would I handle tables in PDFs?

TBD.

## What would I change for production?

TBD.
