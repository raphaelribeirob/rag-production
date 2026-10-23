"""Chunk size / reranker experiment. Every step is resumable and skips finished outputs.

  python tools/retrieval_experiment.py retrieval --config c600_rr --labeled evals/dev_set.jsonl \
      --out DIR/retrieval_c600_rr.json
  LLM_PROVIDER=llamacpp LLAMACPP_MODEL_PATH=... python tools/retrieval_experiment.py generate \
      --config c600_rr --labeled ... --seed 0 --temperature 0.3 --out DIR/answers_c600_rr_s0.jsonl
  python tools/retrieval_experiment.py summarize --dir DIR --base c1200 --new c600_rr --seeds 0 1 2

Configs differ only in chunking and reranking (prompt v1, top-5 context, threshold 0.40).
Each chunk config is ingested into its own table, so the baseline table is never touched.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_production.eval_runner import (  # noqa: E402
    answer_record,
    evaluate,
    latency_summary,
    load_labeled_set,
)
from rag_production.experiment import (  # noqa: E402
    append_jsonl,
    paired_bootstrap,
    per_question_mean,
    read_jsonl,
    run_interval,
    verdict,
)

RR = {"RERANKER": "cross-encoder/ms-marco-MiniLM-L6-v2"}
C1200, C600, C400 = (
    {"CHUNK_SIZE": "1200", "CHUNK_OVERLAP": "0"},
    {"CHUNK_SIZE": "600", "CHUNK_OVERLAP": "100"},
    {"CHUNK_SIZE": "400", "CHUNK_OVERLAP": "80"},
)
CONFIGS = {
    "c1200": C1200,  # baseline (current default)
    "c1200_rr": C1200 | RR,
    "c600": C600,
    "c600_rr": C600 | RR,
    "c400": C400,
    "c400_rr": C400 | RR,
}
JUDGES = {
    "small": "cross-encoder/nli-deberta-v3-small",
    "large": "cross-encoder/nli-deberta-v3-large",
}
RATES = ("supported", "hallucination", "evasive", "paraphrase")


def _setup(config: str):
    from rag_production.chunking import chunk_markdown
    from rag_production.config import load_settings
    from rag_production.embeddings import SentenceTransformerEmbedder
    from rag_production.ingest import ingest_dir
    from rag_production.retrieval import Retriever, get_reranker, table_for
    from rag_production.store import VectorStore

    s = load_settings(dict(os.environ) | CONFIGS[config])
    emb = SentenceTransformerEmbedder(s.embedding_model)
    store = VectorStore(s.database_url, dim=emb.dim, table=table_for(s, "chunks_eval"))
    store.migrate()
    docs = sorted(Path(s.corpus_dir).glob("*.md"))
    expected = sum(
        len(chunk_markdown(d.name, d.read_text("utf-8"), s.chunk_size, s.chunk_overlap))
        for d in docs
    )
    if store.count() != expected:  # first use of this config (or a stale table): re-ingest
        store.clear()
        ingest_dir(s.corpus_dir, store, emb, s.chunk_size, s.chunk_overlap)
    info = {"config": config, "table": store.table, "chunks": store.count()}
    info |= {"chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap}
    info |= {"reranker": s.reranker or None, "reranker_revision": s.reranker_revision}
    info |= {"rerank_candidates": s.rerank_candidates if s.reranker else None}
    return s, Retriever(store, emb, get_reranker(s), s.rerank_candidates), info


def cmd_retrieval(a) -> None:
    s, retriever, info = _setup(a.config)
    labeled = load_labeled_set(a.labeled)
    ms: dict[str, list[float]] = {}
    final: dict[str, list[tuple[str, float]]] = {}
    for row in labeled:
        r = retriever.retrieve(row["question"], k=5)
        final[row["question"]] = [(h.chunk_id, h.score) for h in r.hits]
        for k, v in r.ms.items():
            ms.setdefault(k, []).append(v)
    std = evaluate(labeled, lambda q, n: retriever.ranked(q, n * 4), 5, s.abstain_threshold)
    ctx = evaluate(labeled, lambda q, n: final[q], 5, s.abstain_threshold)
    out = info | {"labeled": a.labeled, "standard": std}
    out["context_top5_chunks"] = {k: ctx[k] for k in ("answerable", "by_category", "misses")}
    out["latency"] = {k: latency_summary(v) for k, v in ms.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


def cmd_generate(a) -> None:
    from rag_production.providers import get_provider

    meta_path = Path(a.out).with_suffix(".meta.json")
    os.environ |= {"LLM_SEED": str(a.seed), "LLM_TEMPERATURE": str(a.temperature)}
    s, retriever, info = _setup(a.config)
    provider = get_provider(s)
    assert provider is not None, "set LLM_PROVIDER (llamacpp) for generation"
    done = {r["id"] for r in read_jsonl(a.out)}
    for row in load_labeled_set(a.labeled):
        if not row["relevant_doc_ids"] or row["id"] in done:
            continue
        t0 = time.perf_counter()
        r = retriever.retrieve(row["question"], k=5)
        ids, texts = [h.chunk_id for h in r.hits], [h.text for h in r.hits]
        stage = dict(r.ms)
        abstained = not r.hits or r.top_cosine < s.abstain_threshold
        t1 = time.perf_counter()
        answer = "" if abstained else provider.generate(row["question"], texts)
        stage["generate"] = round((time.perf_counter() - t1) * 1000, 1)
        stage["total"] = round((time.perf_counter() - t0) * 1000, 1)
        rec = answer_record(row, ids, texts, answer, stage["generate"])
        append_jsonl(a.out, rec | {"abstained": abstained, "stage_ms": stage})
        print(row["id"], stage, flush=True)
    rows = read_jsonl(a.out)
    lat = {k: latency_summary([r["stage_ms"][k] for r in rows if k in r["stage_ms"]])
           for k in ("embed", "retrieve", "rerank", "generate", "total")}  # fmt: skip
    meta = info | {"provider": provider.name, "prompt": s.llm_prompt, "seed": a.seed}
    meta |= {"temperature": a.temperature, "labeled": a.labeled, "n": len(rows)}
    meta |= {"abstained": sum(r["abstained"] for r in rows), "latency": lat}
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")


def _arm(d: Path, config: str, seeds, judge: str) -> dict:
    reps = [json.loads((d / f"judge_{judge}_{config}_s{s}.json").read_text()) for s in seeds]
    out = {"faithfulness": run_interval([r["faithfulness"]["faithfulness"] for r in reps])}
    out["negative_control"] = run_interval([r["faithfulness"]["negative_control"] for r in reps])
    for k in RATES:
        out[k] = run_interval([r["taxonomy"]["rates"][k] for r in reps])
    out["per_q"] = per_question_mean([r["items"] for r in reps])
    return out


def cmd_summarize(a) -> None:
    d = Path(a.dir)
    res: dict = {"base": a.base, "new": a.new, "seeds": a.seeds, "judges": {}}
    for j in JUDGES:
        base, new = _arm(d, a.base, a.seeds, j), _arm(d, a.new, a.seeds, j)
        boot = paired_bootstrap(base.pop("per_q"), new.pop("per_q"))
        res["judges"][j] = {"base": base, "new": new, "bootstrap": boot}
    res["default_changes"] = verdict(
        {j: {"bootstrap": v["bootstrap"], "base": v["base"]["faithfulness"],
             "new": v["new"]["faithfulness"]} for j, v in res["judges"].items()}
    )  # fmt: skip
    for arm in (a.base, a.new):
        metas = [json.loads((d / f"answers_{arm}_s{s}.meta.json").read_text()) for s in a.seeds]
        res[f"latency_{arm}"] = {
            k: run_interval([m["latency"][k]["mean_ms"] for m in metas])
            for k in ("embed", "retrieve", "rerank", "generate", "total")
            if all(m["latency"].get(k) for m in metas)
        }
    body = ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in res.items())
    Path(a.out).write_text("{\n" + body + "\n}\n")  # compact: keeps the file under 350 lines
    print(json.dumps(res, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("retrieval", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--config", choices=sorted(CONFIGS), required=True)
        p.add_argument("--labeled", required=True)
        p.add_argument("--out", required=True)
        if name == "generate":
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--temperature", type=float, required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--dir", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "retrieval" and Path(a.out).exists():
        print(f"skip: {a.out} exists")
        return 0
    if a.cmd == "generate" and Path(a.out).with_suffix(".meta.json").exists():
        print(f"skip: {a.out} complete")
        return 0
    {"retrieval": cmd_retrieval, "generate": cmd_generate, "summarize": cmd_summarize}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
