# Interview notes

Fill in short answers as you build (spec section 16). Write down what you *measured*, not what you expect.

## Why this chunk size and overlap? What happened when I changed them?

Starting point: 800 chars / 150 overlap (spec default). Chunking is done per page so every chunk
maps to exactly one page number, which keeps citations exact. Results of E2 (400 / 800 / 1200): TBD.

## How do I stop the model hallucinating, and how did I measure it?

TBD (grounded prompt + refusal rule; RAGAS faithfulness; refusal accuracy on out-of-scope questions).

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
