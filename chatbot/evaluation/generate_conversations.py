"""Generate synthetic multi-turn conversations (the ground truth for evaluation).

Each conversation is seeded with 2-3 FAQ records from the same section. The LLM
writes one user turn per record; every turn after the first must be a follow-up
that can't be understood without the earlier turns ("what about the deadline
for that?"). That is exactly what single-turn RAG gets wrong.

    python -m evaluation.generate_conversations --n 30 --course llm-zoomcamp
"""

import argparse
import random
from collections import defaultdict

from evaluation.common import CONVERSATIONS_PATH, JUDGE_MODEL, llm_json, write_jsonl
from faqbot.data import load_documents
from faqbot.llm import get_groq_client

SYSTEM = """
You simulate a student chatting with a course assistant. You get FAQ records,
in order. Write one user turn per record, in the same order, as a natural conversation.

Rules:
- Turn 1 is a standalone question answerable from record 1.
- Every later turn is a FOLLOW-UP answerable from its own record, phrased so it
  depends on the earlier turns: use pronouns ("it", "that"), ellipsis
  ("and for the homework?"), or references ("you mentioned X...").
  It must NOT be fully understandable on its own: leave out the main topic
  keyword that the earlier turns already established.
  Bad (standalone):  "How long do I have for the peer reviews of the capstone?"
  Good (follow-up):  "ok and after I submit it, how long do I have for the reviews?"
- Keep follow-ups short (under 20 words).
- Don't copy the FAQ wording; paraphrase like a real student would.
- reference_answer: a short correct answer based only on that turn's record.

Return JSON: {"turns": [{"user": "...", "doc_id": "<record id>", "reference_answer": "..."}]}
""".strip()


def format_records(docs):
    return "\n\n".join(
        f"RECORD {i} (id={d['id']})\nsection: {d['section']}\nQ: {d['question']}\nA: {d['answer']}"
        for i, d in enumerate(docs, 1)
    )


def sample_groups(documents, course, n, rng):
    by_section = defaultdict(list)
    for d in documents:
        if d["course"] == course:
            by_section[d["section"]].append(d)
    sections = [docs for docs in by_section.values() if len(docs) >= 2]

    groups = []
    for _ in range(n):
        docs = rng.choice(sections)
        groups.append(rng.sample(docs, min(len(docs), rng.choice([2, 3]))))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of conversations")
    parser.add_argument("--course", default="llm-zoomcamp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    client = get_groq_client(max_retries=6)
    groups = sample_groups(load_documents(), args.course, args.n, random.Random(args.seed))

    conversations = []
    for i, docs in enumerate(groups):
        try:
            out = llm_json(client, JUDGE_MODEL, SYSTEM, format_records(docs))
        except Exception as e:  # malformed JSON, rate limit after retries, ...
            print(f"[{i}] skipped: {e}")
            continue

        valid_ids = {d["id"] for d in docs}
        turns = [t for t in out.get("turns", []) if t.get("doc_id") in valid_ids and t.get("user")]
        if len(turns) < 2:
            print(f"[{i}] skipped: model returned {len(turns)} valid turns")
            continue

        conversations.append({
            "id": f"synthetic-{i:03d}",
            "course": args.course,
            "type": "synthetic",
            "turns": turns,
        })
        print(f"[{i}] {turns[0]['user']}  ->  {turns[1]['user']}")

    write_jsonl(CONVERSATIONS_PATH, conversations)
    print(f"\nSaved {len(conversations)} conversations to {CONVERSATIONS_PATH}")
    print("Review a few by hand: synthetic data is only useful if it looks like real users.")


if __name__ == "__main__":
    main()
