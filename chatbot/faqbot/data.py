"""Load the bundled FAQ snapshot (data/) and build the search indexes once per process."""

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from faqbot.config import COURSES_PATH, DOCUMENTS_PATH, EMBEDDINGS_META_PATH, EMBEDDINGS_PATH
from faqbot.search import Index
from faqbot.vector_search import VectorIndex

log = logging.getLogger(__name__)


def load_documents(path: Path = DOCUMENTS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_courses(path: Path = COURSES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(documents: list[dict]) -> Index:
    return Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
    ).fit(documents)


@lru_cache(maxsize=1)
def get_index() -> Index:
    return build_index(load_documents())


def load_embeddings(documents: list[dict]) -> np.ndarray:
    """Load data/embeddings.npy, checking it was built from these exact documents."""
    meta = json.loads(EMBEDDINGS_META_PATH.read_text(encoding="utf-8"))
    if meta["ids"] != [d["id"] for d in documents]:
        raise ValueError(
            "data/embeddings.npy is out of date with data/documents.json. "
            "Run: uv run python chatbot/scripts/build_embeddings.py"
        )
    return np.load(EMBEDDINGS_PATH)


@lru_cache(maxsize=1)
def get_vector_index() -> VectorIndex:
    documents = load_documents()
    return VectorIndex(keyword_fields=["course"]).fit(load_embeddings(documents), documents)


@lru_cache(maxsize=1)
def get_encoder():
    # Imported here so BM25-only use never loads onnxruntime.
    from faqbot.embeddings import MiniLMEncoder
    from faqbot.model_files import ensure_model_files

    return MiniLMEncoder(model_dir=ensure_model_files())


def load_vector_search() -> tuple[VectorIndex, object] | None:
    """Vector index + query encoder, or None if they can't be loaded (the bot then uses BM25)."""
    try:
        return get_vector_index(), get_encoder()
    except Exception:
        log.exception("Vector search unavailable; falling back to BM25 only")
        return None
