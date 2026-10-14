from faqbot.search import Index, tokenize


def test_tokenize_lowercases_and_drops_stop_words():
    assert tokenize("Can I still JOIN the course?") == ["still", "join", "course"]


def test_best_match_ranks_first(index):
    results = index.search("certificate", filter_dict={"course": "llm-zoomcamp"})
    assert results[0]["id"] == "cert"


def test_filter_excludes_other_courses(index):
    results = index.search("still join", filter_dict={"course": "ml-zoomcamp"})
    assert [d["id"] for d in results] == ["ml-join"]


def test_num_results_limits_output(index):
    assert len(index.search("course join certificate groq", num_results=2)) == 2


def test_unknown_terms_return_nothing(index):
    assert index.search("kubernetes") == []


def test_boost_changes_ranking():
    docs = [
        {"id": "in-question", "question": "docker error", "answer": "x", "course": "c"},
        {"id": "in-answer", "question": "x", "answer": "docker error", "course": "c"},
    ]
    idx = Index(text_fields=["question", "answer"], keyword_fields=["course"]).fit(docs)
    assert idx.search("docker", boost_dict={"question": 3.0})[0]["id"] == "in-question"
    assert idx.search("docker", boost_dict={"answer": 3.0})[0]["id"] == "in-answer"
