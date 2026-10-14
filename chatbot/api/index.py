"""HTTP API. Vercel serves this file as a Python function; locally, from chatbot/:

    uvicorn api.index:app --reload
"""

import sys
from pathlib import Path
from typing import Literal

APP_ROOT = Path(__file__).resolve().parent.parent
# Vercel runs this file directly, so make the faqbot package importable.
sys.path.insert(0, str(APP_ROOT))

import openai  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from faqbot import ConversationalRAG, create_bot, load_courses  # noqa: E402
from faqbot.config import DEFAULT_COURSE  # noqa: E402

COURSES = load_courses()
COURSE_IDS = {c["id"] for c in COURSES}

app = FastAPI(title="DataTalksClub FAQ Chatbot")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=40)
    course: str = DEFAULT_COURSE


class Source(BaseModel):
    id: str
    section: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    standalone_question: str
    search_mode: str
    sources: list[Source]


_bot: ConversationalRAG | None = None


def get_bot() -> ConversationalRAG:
    global _bot
    if _bot is None:
        _bot = create_bot()
    return _bot


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/warmup")
def warmup():
    """Load the indexes and the embedding model without calling the LLM.

    The page calls this as soon as it opens, so a cold serverless instance starts
    while the user is still typing and the first question is answered at warm speed.
    """
    bot = get_bot()
    return {"status": "warm", "search_mode": bot.search_mode}


@app.get("/api/courses")
def courses():
    return COURSES


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.course not in COURSE_IDS:
        raise HTTPException(status_code=400, detail=f"Unknown course: {req.course}")
    try:
        return get_bot().chat(
            req.message,
            history=[m.model_dump() for m in req.history],
            course=req.course,
        )
    except openai.RateLimitError as e:  # Groq quota (free tier: tokens per minute/day)
        raise HTTPException(
            status_code=429,
            detail="The assistant is busy right now (LLM rate limit). Please try again shortly.",
        ) from e
    except openai.APIError as e:  # Groq unreachable or failing
        raise HTTPException(
            status_code=502, detail="The language model service is unavailable. Please retry."
        ) from e
    except RuntimeError as e:  # missing API key
        raise HTTPException(status_code=500, detail=str(e)) from e


# Locally there is no CDN in front of the app, so serve public/ from here.
# On Vercel, public/ is served by the CDN and only /api/* reaches this function
# (public/ isn't bundled, hence check_dir=False).
app.mount(
    "/", StaticFiles(directory=APP_ROOT / "public", html=True, check_dir=False), name="public"
)
