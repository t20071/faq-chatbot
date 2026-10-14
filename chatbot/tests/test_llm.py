from types import SimpleNamespace

import httpx
import openai
import pytest

from faqbot.llm import chat_completion_with_fallback, get_groq_client
from faqbot.rag import ConversationalRAG
from tests.conftest import FakeLLM

MODELS = ["primary", "backup"]
MESSAGES = [{"role": "user", "content": "hi"}]


class QuotaLLM(FakeLLM):
    """Rate-limits the `exhausted` models and answers with the others."""

    def __init__(self, exhausted):
        super().__init__()
        self.exhausted = set(exhausted)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        model = kwargs["model"]
        if model in self.exhausted:
            request = httpx.Request("POST", "https://api.groq.com")
            response = httpx.Response(429, request=request)
            raise openai.RateLimitError("limit", response=response, body=None)
        message = SimpleNamespace(content=f"answer from {model}")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_falls_back_when_first_model_is_rate_limited():
    llm = QuotaLLM(exhausted={"primary"})
    assert chat_completion_with_fallback(llm, MODELS, MESSAGES) == "answer from backup"
    assert [c["model"] for c in llm.calls] == MODELS


def test_first_model_used_when_available():
    llm = QuotaLLM(exhausted=set())
    assert chat_completion_with_fallback(llm, MODELS, MESSAGES) == "answer from primary"
    assert len(llm.calls) == 1


def test_raises_when_every_model_is_rate_limited():
    with pytest.raises(openai.RateLimitError):
        chat_completion_with_fallback(QuotaLLM(exhausted=MODELS), MODELS, MESSAGES)


def test_bot_answers_via_fallback(index):
    bot = ConversationalRAG(
        index, QuotaLLM(exhausted={"primary"}), search_mode="bm25",
        chat_model="primary", fallback_models=["backup"],
    )
    assert bot.chat("Can I still join?")["answer"] == "answer from backup"


def test_web_client_never_sleeps_on_rate_limits(monkeypatch):
    # Retrying would wait out Groq's Retry-After (~60 s) and exceed Vercel's 30 s limit.
    monkeypatch.setenv("GROQ_API_KEY", "test")
    client = get_groq_client()
    assert client.max_retries == 0
    assert client.timeout <= 15
