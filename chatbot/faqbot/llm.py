"""Groq client (OpenAI-compatible API)."""

import os

from openai import OpenAI

from faqbot.config import GROQ_BASE_URL


def get_groq_client(max_retries: int = 2) -> OpenAI:
    # OPEN_AI_API_KEY is the variable name used before the project was reorganised.
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, max_retries=max_retries)


def chat_completion(client: OpenAI, model: str, messages: list[dict], **kwargs) -> str:
    if model.startswith("openai/gpt-oss"):
        # Groq's gpt-oss models spend tokens on reasoning; keep it short for latency.
        kwargs.setdefault("reasoning_effort", "low")
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return (response.choices[0].message.content or "").strip()
