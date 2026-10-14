import pytest
from fastapi.testclient import TestClient

import api.index as api
from faqbot.rag import ConversationalRAG
from tests.conftest import FakeLLM


@pytest.fixture
def client(index, monkeypatch):
    bot = ConversationalRAG(index, FakeLLM(["Yes, you can."]), search_mode="bm25")
    monkeypatch.setattr(api, "_bot", bot)
    return TestClient(api.app)


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_courses_lists_bundled_courses(client):
    ids = {c["id"] for c in client.get("/api/courses").json()}
    assert "llm-zoomcamp" in ids


def test_chat_returns_answer_and_sources_without_context(client):
    r = client.post("/api/chat", json={"message": "Can I still join?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Yes, you can."
    assert body["sources"][0]["id"] == "join"
    assert "context" not in body


def test_unknown_course_is_rejected(client):
    r = client.post("/api/chat", json={"message": "hi", "course": "nope"})
    assert r.status_code == 400


def test_empty_message_is_rejected(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Course FAQ Assistant" in r.text


def test_warmup_reports_search_mode(client):
    assert client.get("/api/warmup").json() == {"status": "warm", "search_mode": "bm25"}


def test_static_course_list_matches_api(client):
    assert client.get("/courses.json").json() == client.get("/api/courses").json()


def test_rate_limit_becomes_friendly_429(index, monkeypatch):
    import httpx
    import openai

    class RateLimitedLLM(FakeLLM):
        def _create(self, **kwargs):
            request = httpx.Request("POST", "https://api.groq.com")
            raise openai.RateLimitError(
                "limit", response=httpx.Response(429, request=request), body=None
            )

    bot = ConversationalRAG(index, RateLimitedLLM(), search_mode="bm25")
    monkeypatch.setattr(api, "_bot", bot)
    r = TestClient(api.app).post("/api/chat", json={"message": "Can I still join?"})
    assert r.status_code == 429
    assert "try again" in r.json()["detail"]
