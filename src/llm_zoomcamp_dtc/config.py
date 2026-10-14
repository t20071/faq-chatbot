"""Project paths, settings and the Groq client factory."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "faq.db"

load_dotenv(PROJECT_ROOT / ".env")

FAQ_URL = "https://datatalks.club/faq"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")


def get_groq_client() -> OpenAI:
    # OPEN_AI_API_KEY is the variable name used before the project was reorganised.
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        raise RuntimeError(f"GROQ_API_KEY is not set. Copy .env.example to {PROJECT_ROOT / '.env'}")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
