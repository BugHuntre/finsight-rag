# FinSight RAG

Ask plain-English questions about Indian company annual reports and earnings calls and get
answers grounded in the documents, with citations (company, document, page).

> Work in progress. Full requirements: `FinSight_RAG_Project_Requirements.pdf`.

## Status

- [x] Phase 1: setup (structure, config, CI stub)
- [x] Phase 2: ingestion (PDF -> clean -> chunk -> embed -> ChromaDB, hash-based re-run safety)
- [x] Phase 3: query pipeline (retriever, grounded prompt, LLM via LiteLLM, citation mapping,
      not-found handling via a similarity cutoff)
- [x] Phase 4: FastAPI backend (`/health`, `/query`, `/documents`, `/ingest`) + Streamlit UI
- [x] Phase 5: evaluation (50-question golden set, hit rate@k, MRR, refusal accuracy,
      faithfulness/answer relevancy, experiments E1-E3)
- [ ] Phase 6: Docker, deployment, demo

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

Put PDFs in `data/raw/`, named `<Company>_<FY####>_<annual_report|transcript>.pdf`
(e.g. `Infosys_FY2025_annual_report.pdf`, `HDFC-Bank_FY2025_transcript.pdf`). Alternatively
fill in `scripts/sources.csv` with links and run `python scripts/download_data.py`.

```powershell
python -m finsight.ingest --path data/raw
```

Settings (chunk size, overlap, top-k, models) live in `config.yaml`; override any of them with
`FINSIGHT_<KEY>` environment variables or `.env`.

### Ask a question

Local dev uses [Ollama](https://ollama.com) (free, offline): install it, then
`ollama pull qwen2.5:3b` (fits a 4GB GPU). The deployed demo instead sets
`models.llm_model` to a hosted model, e.g. `groq/llama-3.1-8b-instant`, with
`GROQ_API_KEY` in `.env` (free hosting can't run Ollama).

```powershell
python -m finsight.pipeline "What was TCS's revenue in FY2026?" --company TCS
```

Answers are grounded only in retrieved chunks and cite them inline (`[1]`, `[2]`); a
question with no sufficiently similar chunk in the index (cosine distance above
`retrieval.max_distance` in `config.yaml`) gets a "not found" answer without calling
the LLM at all, rather than relying on the LLM to notice and refuse on its own - a
real failure observed in testing with qwen2.5:3b (see `docs/notes.md`).

### Run the API and UI

```powershell
uvicorn finsight.api:app --reload          # Swagger docs at /docs
streamlit run app/streamlit_app.py         # calls the API above (set FINSIGHT_API_URL if not on :8000)
```

`POST /ingest` is disabled by default (`api.enable_ingest_endpoint: false` in `config.yaml`) -
the spec calls for it to be off in the public demo, since anyone could otherwise trigger a
full re-ingestion over the network.

## Evaluation

```powershell
python eval/run_eval.py --label E1_baseline                                  # full: generation + judge
python eval/run_eval.py --label E3_topk4 --top-k 4 --retrieval-only          # retrieval metrics only, fast
python eval/run_eval.py --label E2_chunk400 --collection e2_chunk400 --chunk-size-label 400 --retrieval-only
```

50 questions in `eval/golden_set.jsonl` (20 factual, 15 explanation, 10 comparison, 5 out-of-scope),
built by reading real chunks from the 5 ingested reports and verifying every fact and page number
against them - not written from memory. **Retrieval runs without the golden set's company filter on
purpose**: it measures whether the system finds the right answer on its own semantic relevance
across the whole corpus, which is the harder, more realistic test (a user won't always pick a
company filter first). RAGAS's faithfulness/answer-relevancy metrics are reimplemented directly via
the same LiteLLM call already used for generation, not the `ragas` package - it fails to import in
this environment over a broken transitive dependency (`ragas.llms.base` unconditionally imports a
class recent `langchain-community` removed); see `docs/notes.md`.

| Experiment | Config | Hit rate@k | MRR | Refusal acc. | Faithfulness | Ans. relevancy |
|---|---|---|---|---|---|---|
| **E1 baseline** | chunk 800/150, top-k 8 | 0.58 | 0.47 | 1.00 (5/5) | 0.81 (n=45) | 0.66 (n=39) |
| E2 chunk 400 | chunk 400/150, top-k 8 | 0.56 | 0.41 | 1.00 | - | - |
| E2 chunk 1200 | chunk 1200/150, top-k 8 | **0.62** | 0.42 | 1.00 | - | - |
| E3 top-k 4 | chunk 800/150, top-k 4 | 0.49 | 0.46 | 1.00 | - | - |

(E2/E3 rows are retrieval-only - no generation or judge call - so their latency isn't comparable to
E1's; full numbers, per-question detail and latency are in `eval/results/`.)

### Key findings

- **Blind retrieval (no company filter) is the single biggest factor below target**, not chunk size
  or top-k. Hit rate@8 (0.58) is well under the spec's 0.80 target, but ad hoc testing during Phase 2/3
  with a company filter applied consistently found the right page. FR5's company filter isn't a UI
  nicety - it's the main retrieval-quality lever this measurement actually found.
- **Chunk size 1200 wins on hit rate but not on MRR** (0.42 vs baseline's 0.47): a bigger chunk is
  more likely to *contain* the right page somewhere in the top-k, but isn't a more precise semantic
  match, so it doesn't rank higher once found - sometimes lower. Chunk size 400 was worse on both.
- **top-k 4 vs 8 is the expected recall/precision trade-off**: hit rate drops (0.49 vs 0.58) but MRR
  barely moves (0.46 vs 0.47), since MRR is dominated by whether the right chunk appears at all in
  the very top ranks, which a smaller k mostly doesn't change once past that point.
- **Refusal accuracy is 100% (5/5)**, but specifically *because* of the retrieval-side distance
  cutoff added in Phase 3 - the LLM's own refusal instruction had already failed once in earlier ad
  hoc testing (it hallucinated an answer to "what is the capital of France?"). This is refusal working
  as designed, not the LLM being reliably well-behaved on its own.
- **Faithfulness (0.81) and answer relevancy (0.66) are below target (0.85), with a real caveat**:
  this uses qwen2.5:3b as its own judge, which the spec itself recommends against ("use the Gemini or
  Groq free tier") - I don't have an API key for a stronger, independent judge. Treat these as a rough
  signal, not a validated benchmark; 6 of 45 relevancy judge calls returned unparseable output and were
  excluded rather than scored as 0 (a small local model doesn't reliably follow a JSON-only instruction).
- **The most important finding, found by reading the answers, not from a metric**: on 2 of 10
  comparison questions, the model gave a confidently-cited but factually *wrong* answer by mixing up
  companies. Asked about TCS's profit growth, it answered with Infosys's real, correctly-stated
  figures (₹26,713cr → ₹29,440cr) misattributed to TCS - because blind retrieval let an Infosys chunk
  into the context, and the small model attributed a real number to the wrong company while still
  producing a plausible-looking citation. The faithfulness judge only gave this partial credit (0.5)
  rather than catching the misattribution outright, since it checks whether a claim is *supported by
  the given context*, not whether that context is about the right company - a real blind spot in the
  metric, not just the model. In production, the company filter should probably be required or
  auto-detected from the question, not left optional, specifically to prevent this.
- **Hit rate@k likely under-counts real success**: several "misses" (e.g. HDFC Bank's attrition rate,
  Capital Adequacy Ratio, Gross NPA questions) retrieved a page different from the golden set's, yet
  the generated answer was still correct and specific - annual reports often restate the same figure
  in more than one section (MD&A commentary plus a ratios/highlights summary). The golden set records
  one canonical page per fact; a stricter evaluation would accept any page that genuinely supports the
  answer, not just the one originally cited.

**Configuration chosen for now: unchanged (chunk 800/150, top-k 8).** Chunk size 1200 measured a
real hit-rate improvement, but it cost MRR, the existing live index is already built at 800, and this
signal comes from a self-graded, unaudited golden set - not a strong enough basis to re-ingest
everything on its own. I'm recording the recommendation here rather than acting on it immediately:
revisit with E4 (reranking) and E5 (hybrid search), and against a hosted judge, before deciding.

### Not run (stretch, per the spec's own "if time" framing)

E4 (cross-encoder reranking) and E5 (hybrid BM25 + vector search) - FR10, both stretch goals. Given
the company-filter finding above, my guess is E5 (hybrid/keyword matching) would help most with
exact company names and numbers specifically, but that's a guess, not a measurement - it isn't in
this table because I didn't run it.

## Tests

```powershell
ruff check .
pytest --cov=finsight
```
