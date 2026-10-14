"""Assemble a ready-to-use chatbot from the bundled data."""

import logging

from openai import OpenAI

from faqbot.config import SEARCH_MODE
from faqbot.data import get_index, load_vector_search
from faqbot.llm import get_groq_client
from faqbot.rag import ConversationalRAG

log = logging.getLogger(__name__)


def create_bot(
    client: OpenAI | None = None, search_mode: str = SEARCH_MODE, **kwargs
) -> ConversationalRAG:
    """Build the bot; if vector search can't load, degrade to BM25 instead of failing."""
    vector_index = encoder = None
    if search_mode != "bm25":
        loaded = load_vector_search()
        if loaded:
            vector_index, encoder = loaded
        else:
            log.warning("Using search_mode='bm25' instead of %r", search_mode)
            search_mode = "bm25"
    return ConversationalRAG(
        get_index(),
        client or get_groq_client(),
        vector_index=vector_index,
        encoder=encoder,
        search_mode=search_mode,
        **kwargs,
    )
