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
- [ ] Phase 5: evaluation (golden set, hit rate@k, MRR, RAGAS)
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

## Tests

```powershell
ruff check .
pytest --cov=finsight
```
