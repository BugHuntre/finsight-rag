"""Embedding model wrapper shared by ingestion and retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence

# BGE models expect this instruction on the *query* side only (not on documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


class Embedder:
    """Lazy sentence-transformers wrapper producing L2-normalised vectors."""

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        vecs = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, question: str) -> list[float]:
        """Embed a user question for retrieval."""
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self.embed_documents([prefix + question])[0]
