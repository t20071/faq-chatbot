"""Retrieval quality per turn: hit rate@5 and MRR for every retriever × query strategy.

Retrievers:
  bm25     keyword search
  vector   semantic search (all-MiniLM-L6-v2 embeddings)
  hybrid   both, fused with reciprocal rank fusion
Queries:
  raw            only the latest user message (single-turn RAG)
  raw+rewrite    the latest message AND its LLM-rewritten standalone version

"hybrid:raw" is what the chatbot does by default (REWRITE_QUERIES=false).

Uses "teacher forcing": the history for turn N is the ground-truth user turns and
reference answers, so every strategy sees the same conversation.

    python -m evaluation.evaluate_retrieval
"""

import json
import sys

from evaluation.common import RESULTS_DIR, load_conversations, mean
from faqbot.bot import create_bot
from faqbot.llm import get_groq_client

RETRIEVERS = ["bm25", "vector", "hybrid"]
QUERY_SETS = {"raw": ["raw"], "raw+rewrite": ["raw", "rewrite"]}
STRATEGIES = [f"{r}:{q}" for r in RETRIEVERS for q in QUERY_SETS]


def rank_of(doc_id, results):
    for i, doc in enumerate(results, 1):
        if doc["id"] == doc_id:
            return i
    return None


def main():
    # rewrite=True: this script compares searching with and without the rewrite.
    bot = create_bot(get_groq_client(max_retries=6), search_mode="hybrid", rewrite=True)
    if bot.search_mode != "hybrid":
        sys.exit("Vector search failed to load; see the log above.")
    rows = []

    for conv in load_conversations():
        history = []
        for turn_no, turn in enumerate(conv["turns"], 1):
            question = turn["user"]
            if turn.get("doc_id"):
                queries = {"raw": question, "rewrite": bot.standalone_question(question, history)}
                row = {"conversation": conv["id"], "turn": turn_no, "doc_id": turn["doc_id"],
                       "rewritten_query": queries["rewrite"]}
                for retriever in RETRIEVERS:
                    for name, keys in QUERY_SETS.items():
                        results = bot.fused_search(
                            [queries[k] for k in keys], conv["course"], mode=retriever
                        )
                        row[f"{retriever}:{name}"] = rank_of(turn["doc_id"], results)
                rows.append(row)

            history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": turn.get("reference_answer", "")},
            ]
        print(".", end="", flush=True)
    print()

    def summarize(subset):
        return {
            s: {
                "hit_rate": mean([r[s] is not None for r in subset]),
                "mrr": mean([1 / r[s] if r[s] else 0 for r in subset]),
            }
            for s in STRATEGIES
        }

    summary = {
        "n_turns": len(rows),
        "first_turn": summarize([r for r in rows if r["turn"] == 1]),
        "follow_up_turns": summarize([r for r in rows if r["turn"] > 1]),
        "all_turns": summarize(rows),
    }

    print(f"{'subset':<16} {'strategy':<20} {'hit@5':>7} {'MRR':>7}")
    for subset in ["first_turn", "follow_up_turns", "all_turns"]:
        for s in STRATEGIES:
            m = summary[subset][s]
            if m["hit_rate"] is not None:
                print(f"{subset:<16} {s:<20} {m['hit_rate']:>7.3f} {m['mrr']:>7.3f}")
        print()

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "retrieval.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(f"Details: {out}")


if __name__ == "__main__":
    main()
