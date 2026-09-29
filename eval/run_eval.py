"""Run the golden-set evaluation (spec section 10): retrieval quality, refusal
accuracy, and (unless --retrieval-only) generation quality via an LLM judge.

Usage::

    python eval/run_eval.py --label E1_baseline
    python eval/run_eval.py --label E3_topk4 --top-k 4 --retrieval-only
    python eval/run_eval.py --label E2_chunk400 --collection e2_chunk400 --retrieval-only

Retrieval and generation are run WITHOUT the golden set's company/fiscal_year
filter, on purpose: this measures whether the system finds the right answer on
its own semantic relevance across the whole corpus, which is the harder and more
realistic test (a Streamlit user won't always pick a company filter first).

Results are written to eval/results/<label>.json (full per-question detail) and
appended as one row to eval/results/summary.csv (for the README's results table).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from finsight.config import load_config
from finsight.embeddings import Embedder
from finsight.generator import NOT_FOUND_MESSAGE, generate_answer
from finsight.ingest import get_collection
from finsight.prompts import format_context
from finsight.retriever import retrieve

# Python adds a directly-run script's own directory to sys.path automatically,
# and tests get the same via pyproject.toml's `pythonpath = ["src", "eval"]`, so
# these siblings import cleanly either way without extra path manipulation.
from llm_judge import answer_relevancy, faithfulness
from metrics import hit_rate_at_k, mean, percentile, reciprocal_rank

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_GOLDEN_SET = Path(__file__).parent / "golden_set.jsonl"


def load_golden_set(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _page(metadata: dict) -> int:
    return metadata.get("printed_page") or metadata["page"]


def evaluate(
    golden_set: list[dict],
    cfg,
    collection,
    embedder: Embedder,
    retrieval_only: bool,
    judge_model: str,
    relevancy_n: int,
) -> dict:
    per_question = []
    for item in golden_set:
        t0 = time.time()
        chunks = retrieve(item["question"], cfg, collection, embedder)
        retrieved_pages = [_page(c.metadata) for c in chunks]
        row: dict = {"id": item["id"], "type": item["type"]}

        if item["type"] == "out_of_scope":
            result = generate_answer(item["question"], chunks, cfg.llm_model)
            row["refused_correctly"] = result.answer.strip() == NOT_FOUND_MESSAGE
            row["latency_ms"] = int((time.time() - t0) * 1000)
        else:
            row["hit"] = hit_rate_at_k(retrieved_pages, item["pages"])
            row["reciprocal_rank"] = reciprocal_rank(retrieved_pages, item["pages"])
            if not retrieval_only:
                result = generate_answer(item["question"], chunks, cfg.llm_model)
                row["answer"] = result.answer
                row["latency_ms"] = int((time.time() - t0) * 1000)
                if chunks:
                    context = format_context(chunks)
                    row["faithfulness"] = faithfulness(context, result.answer, judge_model)
                    row["answer_relevancy"] = answer_relevancy(
                        item["question"], result.answer, judge_model, embedder, n=relevancy_n
                    )
            else:
                row["latency_ms"] = int((time.time() - t0) * 1000)
        per_question.append(row)
        print(f"  [{item['id']}] {item['type']} -> {row}")

    in_scope = [r for r in per_question if r["type"] != "out_of_scope"]
    out_of_scope = [r for r in per_question if r["type"] == "out_of_scope"]
    faithfulness_scores = [r["faithfulness"] for r in in_scope if r.get("faithfulness") is not None]
    relevancy_scores = [
        r["answer_relevancy"] for r in in_scope if r.get("answer_relevancy") is not None
    ]
    latencies = [r["latency_ms"] for r in per_question]

    summary = {
        "hit_rate_at_k": mean([1.0 if r["hit"] else 0.0 for r in in_scope]),
        "mrr": mean([r["reciprocal_rank"] for r in in_scope]),
        "refusal_accuracy": mean([1.0 if r["refused_correctly"] else 0.0 for r in out_of_scope]),
        "faithfulness": mean(faithfulness_scores) if faithfulness_scores else None,
        "faithfulness_n": len(faithfulness_scores),
        "answer_relevancy": mean(relevancy_scores) if relevancy_scores else None,
        "answer_relevancy_n": len(relevancy_scores),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "n_questions": len(golden_set),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_of_scope),
    }
    return {"summary": summary, "per_question": per_question}


def write_results(label: str, config_desc: dict, result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{label}.json"
    out_path.write_text(
        json.dumps({"label": label, "config": config_desc, **result}, indent=2), encoding="utf-8"
    )

    summary_path = RESULTS_DIR / "summary.csv"
    fields = ["label", *config_desc.keys(), *result["summary"].keys()]
    row = {"label": label, **config_desc, **result["summary"]}
    is_new = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nWrote {out_path} and appended a row to {summary_path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--label", required=True, help="Experiment label, e.g. E1_baseline")
    p.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--collection", type=str, default=None)
    p.add_argument("--index-dir", type=Path, default=None)
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation/faithfulness/relevancy (for chunk-size/top-k experiments)",
    )
    p.add_argument("--judge-model", type=str, default=None, help="Defaults to cfg.llm_model")
    p.add_argument("--relevancy-n", type=int, default=3)
    p.add_argument(
        "--chunk-size-label",
        type=int,
        default=None,
        help="Chunk size to RECORD in results, e.g. when --collection points at an index "
        "built with a different chunk size than the current config.yaml - chunk size only "
        "matters at ingestion time, so querying can't discover it from the collection itself. "
        "Defaults to cfg.chunk_size, which is only correct when querying the config's own "
        "default collection.",
    )
    args = p.parse_args(argv)

    cfg = load_config(
        top_k=args.top_k,
        collection=args.collection,
        index_dir=args.index_dir.resolve() if args.index_dir else None,
        max_distance=args.max_distance,
    )
    golden_set = load_golden_set(args.golden_set)
    collection = get_collection(cfg)
    embedder = Embedder(cfg.embedding_model, cfg.embed_batch_size)
    judge_model = args.judge_model or cfg.llm_model

    print(f"Running {args.label}: {len(golden_set)} questions against collection "
          f"'{cfg.collection}' ({collection.count()} chunks), top_k={cfg.top_k}, "
          f"max_distance={cfg.max_distance}, retrieval_only={args.retrieval_only}")

    result = evaluate(
        golden_set, cfg, collection, embedder, args.retrieval_only, judge_model, args.relevancy_n
    )

    chunk_size = args.chunk_size_label if args.chunk_size_label is not None else cfg.chunk_size
    config_desc = {
        "chunk_size": chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "top_k": cfg.top_k,
        "max_distance": cfg.max_distance,
        "collection": cfg.collection,
        "llm_model": cfg.llm_model,
        "retrieval_only": args.retrieval_only,
    }
    write_results(args.label, config_desc, result)
    print("\nSummary:", json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
