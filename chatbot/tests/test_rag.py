from faqbot.config import MAX_HISTORY_MESSAGES
from faqbot.rag import ConversationalRAG
from tests.conftest import FakeLLM


def test_first_turn_skips_rewriting(index, fake_llm):
    bot = ConversationalRAG(index, fake_llm, search_mode="bm25")
    result = bot.chat("How do I get a certificate?")

    assert len(fake_llm.calls) == 1  # answer only, no rewrite call
    assert result["standalone_question"] == "How do I get a certificate?"
    assert result["sources"][0]["id"] == "cert"


def test_follow_up_is_rewritten_before_search(index):
    llm = FakeLLM(replies=["Can I use Groq instead of OpenAI?", "Yes."])
    bot = ConversationalRAG(index, llm, search_mode="bm25", rewrite=True)
    history = [
        {"role": "user", "content": "Do I need OpenAI?"},
        {"role": "assistant", "content": "No."},
    ]
    result = bot.chat("what about that other one?", history=history)

    assert result["standalone_question"] == "Can I use Groq instead of OpenAI?"
    assert result["sources"][0]["id"] == "groq"
    assert result["answer"] == "Yes."


def test_rewrite_is_off_by_default(index, fake_llm):
    bot = ConversationalRAG(index, fake_llm, search_mode="bm25")
    bot.chat("and that?", history=[{"role": "user", "content": "hi"}])
    assert len(fake_llm.calls) == 1


def test_history_is_trimmed(index, fake_llm):
    history = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    bot = ConversationalRAG(index, fake_llm, search_mode="bm25", rewrite=False)
    bot.chat("certificate?", history=history)

    messages = fake_llm.calls[0]["messages"]
    # system prompt + trimmed history + current question
    assert len(messages) == 1 + MAX_HISTORY_MESSAGES + 1
    assert messages[1]["content"] == f"m{20 - MAX_HISTORY_MESSAGES}"


def test_fused_search_deduplicates(index, fake_llm):
    bot = ConversationalRAG(index, fake_llm, search_mode="bm25")
    results = bot.fused_search(["certificate", "certificate", "join"], "llm-zoomcamp")
    ids = [d["id"] for d in results]
    assert len(ids) == len(set(ids))
    assert {"cert", "join"} <= set(ids)
