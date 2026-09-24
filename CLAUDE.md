Create a CLAUDE.md file in the project root with these standing rules for this project:

- Project: FinSight RAG, a Q&A assistant over Indian company annual reports with
  page-level citations. The spec PDF is the source of truth for scope and structure.
- Python 3.11+, package in src/finsight, config in config.yaml, secrets only in .env
  (never commit keys; keep .env.example updated).
- Hardware: GTX 1650 with 4 GB VRAM. Local LLM is Ollama qwen2.5:3b; deployed LLM
  is Groq free tier. No models larger than 4 GB.
- Every change must keep ruff clean, all tests passing, and coverage above 80%.
  Mock the LLM in tests so CI needs no Ollama or API key.
- Never invent data, URLs, or evaluation numbers. If something is unknown, ask me.
- After each phase: summarise what changed, list any decisions I should know about,
  and make a git commit with a clear message.
- Explain design choices in plain language, because I need to defend them in interviews.