"""Conversational (multi-turn) RAG over the DataTalksClub FAQ."""

from openai import OpenAI

from faqbot.config import (
    CHAT_MODEL,
    DEFAULT_COURSE,
    MAX_HISTORY_MESSAGES,
    NUM_RESULTS,
    REWRITE_MODEL,
    REWRITE_QUERIES,
    SEARCH_MODE,
)
from faqbot.llm import chat_completion
from faqbot.prompts import (
    ANSWER_INSTRUCTIONS,
    REWRITE_INSTRUCTIONS,
    REWRITE_TEMPLATE,
    USER_PROMPT_TEMPLATE,
)
from faqbot.search import Index
from faqbot.vector_search import VectorIndex

SEARCH_MODES = ("hybrid", "bm25", "vector")


def format_history(history: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge rankings: each list adds 1 / (k + rank) to a document's score.

    Uses ranks only, so BM25 scores and cosine similarities never need to be put
    on the same scale. A document found by several lists rises to the top.
    """
    scores, docs = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, 1):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k + rank)
            docs[doc["id"]] = doc
    return [docs[i] for i in sorted(scores, key=scores.get, reverse=True)]


class ConversationalRAG:
    """RAG that handles follow-up questions.

    history: list of {"role": "user" | "assistant", "content": str}, oldest first,
    NOT including the current question.
    """

    def __init__(
        self,
        index: Index,
        client: OpenAI,
        vector_index: VectorIndex | None = None,
        encoder=None,
        search_mode: str = SEARCH_MODE,
        chat_model: str = CHAT_MODEL,
        rewrite_model: str = REWRITE_MODEL,
        rewrite: bool = REWRITE_QUERIES,
        num_results: int = NUM_RESULTS,
    ):
        if search_mode not in SEARCH_MODES:
            raise ValueError(f"search_mode must be one of {SEARCH_MODES}, got {search_mode!r}")
        if search_mode != "bm25" and (vector_index is None or encoder is None):
            raise ValueError(f"search_mode={search_mode!r} needs a vector_index and an encoder")
        self.index = index
        self.client = client
        self.vector_index = vector_index
        self.encoder = encoder  # anything with .encode(list[str]) -> array of unit vectors
        self.search_mode = search_mode
        self.chat_model = chat_model
        self.rewrite_model = rewrite_model
        self.rewrite = rewrite
        self.num_results = num_results

    def standalone_question(self, question: str, history: list[dict]) -> str:
        """Turn a follow-up like "and how much does it cost?" into a searchable query."""
        if not self.rewrite or not history:
            return question
        prompt = REWRITE_TEMPLATE.format(
            history=format_history(history[-MAX_HISTORY_MESSAGES:]), question=question
        )
        rewritten = chat_completion(
            self.client,
            self.rewrite_model,
            [
                {"role": "system", "content": REWRITE_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return rewritten or question

    def search(self, query: str, course: str, num_results: int | None = None) -> list[dict]:
        """BM25 keyword search."""
        return self.index.search(
            query,
            filter_dict={"course": course},
            boost_dict={"question": 2.0, "section": 0.5},
            num_results=num_results or self.num_results,
        )

    def vector_search(
        self, queries: list[str], course: str, num_results: int | None = None
    ) -> list[list[dict]]:
        """Semantic search: one ranked list per query (queries are embedded in one batch)."""
        vectors = self.encoder.encode(queries)
        return [
            self.vector_index.search(
                v, filter_dict={"course": course}, num_results=num_results or self.num_results
            )
            for v in vectors
        ]

    def fused_search(
        self, queries: list[str], course: str, mode: str | None = None
    ) -> list[dict]:
        """Search every query with every retriever of `mode`, then fuse the rankings.

        The raw message keeps the user's exact terms; the rewritten query adds context
        from earlier turns. BM25 matches exact words (error messages, tool names);
        vectors match meaning ("when is it due?" ~ "deadline").
        """
        mode = mode or self.search_mode
        queries = list(dict.fromkeys(q for q in queries if q))
        depth = self.num_results * 2  # look deeper than needed so fusion has candidates

        ranked_lists = []
        if mode in ("bm25", "hybrid"):
            ranked_lists += [self.search(q, course, depth) for q in queries]
        if mode in ("vector", "hybrid"):
            ranked_lists += self.vector_search(queries, course, depth)

        if len(ranked_lists) == 1:
            return ranked_lists[0][: self.num_results]
        return reciprocal_rank_fusion(ranked_lists)[: self.num_results]

    @staticmethod
    def build_context(search_results: list[dict]) -> str:
        return "\n\n".join(
            f"section: {doc['section']}\nQ: {doc['question']}\nA: {doc['answer']}"
            for doc in search_results
        )

    def answer(self, question: str, history: list[dict], search_results: list[dict]) -> str:
        messages = [{"role": "system", "content": ANSWER_INSTRUCTIONS}]
        messages += [
            {"role": m["role"], "content": m["content"]} for m in history[-MAX_HISTORY_MESSAGES:]
        ]
        messages.append({
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                question=question, context=self.build_context(search_results)
            ),
        })
        return chat_completion(self.client, self.chat_model, messages)

    def chat(
        self, question: str, history: list[dict] | None = None, course: str = DEFAULT_COURSE
    ) -> dict:
        history = history or []
        query = self.standalone_question(question, history)
        results = self.fused_search([query, question], course)
        answer = self.answer(question, history, results)
        return {
            "answer": answer,
            "standalone_question": query,
            "search_mode": self.search_mode,
            "sources": [
                {"id": d["id"], "section": d["section"], "question": d["question"]}
                for d in results
            ],
            # Full text is used by the evaluators; the API strips it.
            "context": self.build_context(results),
        }
