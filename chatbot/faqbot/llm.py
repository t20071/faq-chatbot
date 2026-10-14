"""Groq client (OpenAI-compatible API)."""

import logging
import os

import openai
from openai import OpenAI

from faqbot.config import GROQ_BASE_URL, LLM_TIMEOUT_SECONDS

log = logging.getLogger(__name__)


def get_groq_client(max_retries: int = 0, timeout: float = LLM_TIMEOUT_SECONDS) -> OpenAI:
    """Client for the web app: no automatic retries by default.

    On a rate limit, the OpenAI SDK would otherwise sleep for Groq's Retry-After
    (often ~60 s), far past Vercel's 30 s function limit; failing fast lets the app
    try a fallback model or show "busy" instead. Batch jobs (evaluation) pass
    max_retries explicitly to wait it out.
    """
    # OPEN_AI_API_KEY is the variable name used before the project was reorganised.
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return OpenAI(
        api_key=api_key, base_url=GROQ_BASE_URL, max_retries=max_retries, timeout=timeout
    )


def chat_completion(client: OpenAI, model: str, messages: list[dict], **kwargs) -> str:
    if model.startswith("openai/gpt-oss"):
        # Groq's gpt-oss models spend tokens on reasoning; keep it short for latency.
        kwargs.setdefault("reasoning_effort", "low")
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return (response.choices[0].message.content or "").strip()


def chat_completion_with_fallback(
    client: OpenAI, models: list[str], messages: list[dict], **kwargs
) -> str:
    """Try each model in order, moving on when one is rate limited.

    Groq's free-tier limits (tokens per minute/day) are counted per model, so a
    second model keeps the bot answering after the first one's quota runs out.
    """
    for i, model in enumerate(models):
        try:
            return chat_completion(client, model, messages, **kwargs)
        except openai.RateLimitError:
            if i == len(models) - 1:
                raise
            log.warning("%s is rate limited; falling back to %s", model, models[i + 1])
    raise ValueError("models must not be empty")
