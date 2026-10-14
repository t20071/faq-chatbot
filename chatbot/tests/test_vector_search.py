import numpy as np
import pytest

from faqbot.rag import ConversationalRAG, reciprocal_rank_fusion
from faqbot.vector_search import VectorIndex


def test_nearest_vector_ranks_first():
    docs = [{"id": "a", "course": "c"}, {"id": "b", "course": "c"}]
    idx = VectorIndex(keyword_fields=["course"]).fit(np.eye(2, dtype=np.float32), docs)
    assert [d["id"] for d in idx.search(np.array([0.1, 0.9]))] == ["b", "a"]


def test_filter_excludes_other_courses(vector_index, encoder):
    results = vector_index.search(encoder.encode(["can I still join?"])[0],
                                  filter_dict={"course": "ml-zoomcamp"})
    assert [d["id"] for d in results] == ["ml-join"]


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        VectorIndex(keyword_fields=[]).fit(np.zeros((2, 3)), [{"id": "only-one"}])


def test_rrf_rewards_documents_found_by_several_lists():
    a, b, c = ({"id": x} for x in "abc")
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert fused[0]["id"] == "b"


def test_vector_search_finds_synonyms_bm25_misses(index, vector_index, encoder, fake_llm):
    # "due" never appears in the FAQ, "deadline" does: keywords miss it, vectors don't.
    common = dict(vector_index=vector_index, encoder=encoder)
    bm25 = ConversationalRAG(index, fake_llm, search_mode="bm25", **common)
    vector = ConversationalRAG(index, fake_llm, search_mode="vector", **common)

    assert bm25.fused_search(["when is it due"], "llm-zoomcamp") == []
    assert vector.fused_search(["when is it due"], "llm-zoomcamp")[0]["id"] == "join"


def test_hybrid_combines_both_retrievers(index, vector_index, encoder, fake_llm):
    bot = ConversationalRAG(index, fake_llm, vector_index=vector_index, encoder=encoder)
    assert bot.search_mode == "hybrid"
    ids = [d["id"] for d in bot.fused_search(["certificate due"], "llm-zoomcamp")]
    assert {"cert", "join"} <= set(ids)


def test_vector_modes_require_an_encoder(index, fake_llm):
    with pytest.raises(ValueError):
        ConversationalRAG(index, fake_llm, search_mode="hybrid")


def test_create_bot_falls_back_to_bm25(monkeypatch, fake_llm):
    import faqbot.bot as bot_module

    monkeypatch.setattr(bot_module, "load_vector_search", lambda: None)
    assert bot_module.create_bot(fake_llm).search_mode == "bm25"
