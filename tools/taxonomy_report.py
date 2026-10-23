"""Classify saved answers (eval_runner --save-answers) with the fixed taxonomy rules.

  python tools/taxonomy_report.py --answers reports/answers.json --out evals/results/x.json \
      [--labeled evals/labeled_set.jsonl] [--judge cross-encoder/nli-deberta-v3-small] \
      [--backfill] [--hand evals/audit/hand_labels.json] [--sample-out audit.md --sample 20]

--backfill: answers saved before 2026-10-23 have no ids or doc ids; they are matched in order to
the answerable questions and retrieval is re-run (deterministic) to recover the top-5 doc ids;
the recovered context texts must equal the saved ones or the tool stops.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_production.eval_runner import answer_record, load_labeled_set  # noqa: E402
from rag_production.faithfulness import split_sentences  # noqa: E402
from rag_production.taxonomy import agreement, classify_answer, summarize  # noqa: E402


def backfill(items: list[dict], labeled: list[dict]) -> list[dict]:
    from rag_production.config import load_settings
    from rag_production.embeddings import SentenceTransformerEmbedder
    from rag_production.store import VectorStore

    settings = load_settings()
    emb = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(settings.database_url, dim=emb.dim, table="chunks_eval")
    rows = [r for r in labeled if r["relevant_doc_ids"]]
    assert len(rows) == len(items), "answer count != answerable questions (abstentions?)"
    out = []
    for row, it in zip(rows, items, strict=True):
        hits = store.search(emb.embed([row["question"]])[0], k=5)
        texts = [h.text for h in hits]
        if texts != it["contexts"]:
            raise SystemExit(f"{row['id']}: re-retrieved contexts differ from the saved ones")
        out.append(answer_record(row, [h.chunk_id for h in hits], texts, it["answer"], 0.0))
    store.close()
    return out


def write_sample(items: list[dict], n: int, path: str) -> list[str]:
    """Seeded sample of scorable answers for a blind hand audit (no classifier output shown)."""
    ids = sorted(it["id"] for it in items if split_sentences(it["answer"]))
    picked = sorted(random.Random(0).sample(ids, min(n, len(ids))))
    by_id = {it["id"]: it for it in items}
    lines = []
    for i in picked:
        it = by_id[i]
        lines += [f"## {i}: {it['question']}"]
        lines += [f"gold {it['relevant_doc_ids']} top5 {it['doc_ids']}"]
        lines += ["ANSWER: " + it["answer"], ""]
        lines += [f"[{k}] {c}\n" for k, c in enumerate(it["contexts"])]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--labeled", default="evals/labeled_set.jsonl")
    ap.add_argument("--judge", default="cross-encoder/nli-deberta-v3-small")
    ap.add_argument("--out", default="")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--hand", default="", help="JSON {id: label} from the blind hand audit")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--sample-out", default="", help="write the audit sample and exit")
    args = ap.parse_args()

    items = json.loads(Path(args.answers).read_text())
    if args.backfill:
        items = backfill(items, load_labeled_set(args.labeled))
    if args.sample_out:
        print(json.dumps(write_sample(items, args.sample, args.sample_out)))
        return 0

    from rag_production.faithfulness import NLIJudge, evaluate_faithfulness

    judge = NLIJudge(args.judge)
    per = []
    for it in items:
        r = classify_answer(it["answer"], it["contexts"], it["gold_retrieved"], judge)
        per.append({"id": it["id"], "label": r["label"], "score": r["score"]} | {
            "gold_retrieved": it["gold_retrieved"], "answer": it["answer"],
            "sentences": r["sentences"],
        })  # fmt: skip
    report = {
        "answers": args.answers,
        "judge": judge.name,
        "faithfulness": evaluate_faithfulness(items, judge),
        "taxonomy": summarize([p["label"] for p in per]),
    }
    if args.hand:
        hand = json.loads(Path(args.hand).read_text())["hand"]
        auto = {p["id"]: p["label"] for p in per}
        score = {p["id"]: p["score"] for p in per}
        four = {k: v if v != "grounded" else ("supported" if score.get(k) == 1.0 else "paraphrase")
                for k, v in hand.items()}  # fmt: skip
        grounded = {"supported": "grounded", "paraphrase": "grounded"}
        report["hand_agreement"] = {
            "four_class": agreement(four, auto),
            "three_class": agreement(hand, {k: grounded.get(v, v) for k, v in auto.items()}),
        }
    report["items"] = per
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
