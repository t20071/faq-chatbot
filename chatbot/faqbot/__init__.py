"""Multi-turn RAG chatbot over the DataTalksClub course FAQ."""

from faqbot.bot import create_bot
from faqbot.data import get_index, load_courses, load_documents
from faqbot.llm import get_groq_client
from faqbot.rag import ConversationalRAG

__all__ = [
    "ConversationalRAG",
    "create_bot",
    "get_groq_client",
    "get_index",
    "load_courses",
    "load_documents",
]
