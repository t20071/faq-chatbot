"""Paths and settings. Values come from environment variables.

Locally they are read from the project-root .env.
On Vercel they are set under Project Settings → Environment Variables.
"""

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
COURSES_PATH = DATA_DIR / "courses.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_META_PATH = DATA_DIR / "embeddings.json"

# Embedding model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pinned commit
EMBEDDING_MAX_TOKENS = 256  # the model's max_seq_length
MODEL_DIR = APP_ROOT / "models" / "all-MiniLM-L6-v2"
# 8-bit quantised ONNX export from the model repo: 23 MB instead of 90 MB, small enough
# to ship inside the Vercel function (no download on cold start). Document vectors are
# built from this same file, so queries and documents share one encoder.
ONNX_REPO_FILE = "onnx/model_quint8_avx2.onnx"
ONNX_MODEL_FILE = "model_quint8_avx2.onnx"
# What gets embedded for each FAQ record.
EMBEDDING_TEXT_TEMPLATE = "{question}\n{answer}"

try:  # python-dotenv isn't in requirements.txt, so it's absent on Vercel.
    from dotenv import load_dotenv

    load_dotenv(APP_ROOT.parent / ".env")
except ImportError:
    pass

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
# Tried in order when CHAT_MODEL is rate limited (Groq limits are per model).
FALLBACK_MODELS = [
    m.strip() for m in os.getenv("FALLBACK_MODELS", "openai/gpt-oss-120b").split(",") if m.strip()
]
# Per LLM request. Two attempts (primary + fallback) must fit in Vercel's 30 s limit.
LLM_TIMEOUT_SECONDS = 12.0
REWRITE_MODEL = os.getenv("REWRITE_MODEL", CHAT_MODEL)
# Rewriting follow-ups costs an extra LLM call (~1 s) and didn't improve retrieval in
# evaluation (hybrid hit rate 0.945 with or without it), so it's off by default.
# The answer LLM always sees the conversation history either way.
REWRITE_QUERIES = os.getenv("REWRITE_QUERIES", "false").lower() == "true"
DEFAULT_COURSE = "llm-zoomcamp"

# Previous messages (user + assistant) sent along with each request.
MAX_HISTORY_MESSAGES = 6
# FAQ records retrieved per question.
NUM_RESULTS = 5
# "hybrid" (BM25 + vector, fused), "bm25" or "vector".
SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid")
