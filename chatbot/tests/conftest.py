from types import SimpleNamespace

import numpy as np
import pytest

from faqbot.data import build_index
from faqbot.vector_search import VectorIndex

DOCS = [
    {"id": "join", "course": "llm-zoomcamp", "section": "General",
     "question": "I just discovered the course. Can I still join?",
     "answer": "Yes, submit the project before the deadline to get a certificate."},
    {"id": "cert", "course": "llm-zoomcamp", "section": "General",
     "question": "How do I get a certificate?",
     "answer": "Finish the capstone project and the peer reviews."},
    {"id": "groq", "course": "llm-zoomcamp", "section": "Module 1: RAG",
     "question": "Can I use Groq instead of OpenAI?",
     "answer": "Yes, any OpenAI-compatible provider works."},
    {"id": "ml-join", "course": "ml-zoomcamp", "section": "General",
     "question": "Can I still join the ML course?",
     "answer": "Yes."},
]


class FakeLLM:
    """Stands in for the OpenAI client: records requests, returns canned replies."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0) if self.replies else "fake answer"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def index():
    return build_index(DOCS)


@pytest.fixture
def fake_llm():
    return FakeLLM()


class FakeEncoder:
    """Deterministic stand-in for MiniLM: one dimension per concept, synonyms share one.

    Lets tests show what vectors add over keywords: "due" and "deadline" share
    no characters but land on the same dimension.
    """

    CONCEPTS = [
        {"join", "enroll", "late"},
        {"certificate", "certification"},
        {"groq", "openai", "provider"},
        {"deadline", "due"},
    ]

    def encode(self, texts):
        vectors = np.zeros((len(texts), len(self.CONCEPTS)), dtype=np.float32)
        for i, text in enumerate(texts):
            words = set(text.lower().replace("?", " ").replace(".", " ").split())
            for j, concept in enumerate(self.CONCEPTS):
                vectors[i, j] = len(words & concept)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


@pytest.fixture
def encoder():
    return FakeEncoder()


@pytest.fixture
def vector_index(encoder):
    texts = [f"{d['question']}\n{d['answer']}" for d in DOCS]
    return VectorIndex(keyword_fields=["course"]).fit(encoder.encode(texts), DOCS)
