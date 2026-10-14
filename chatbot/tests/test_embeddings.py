"""Checks the real MiniLM ONNX encoder. Skipped until the model has been downloaded
(uv run python chatbot/scripts/build_embeddings.py)."""

import numpy as np
import pytest

from faqbot.config import MODEL_DIR, ONNX_MODEL_FILE

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / ONNX_MODEL_FILE).exists(), reason="MiniLM ONNX model not downloaded"
)


@pytest.fixture(scope="module")
def encoder():
    from faqbot.embeddings import MiniLMEncoder

    return MiniLMEncoder()


def test_vectors_are_384_dim_and_unit_length(encoder):
    vectors = encoder.encode(["Can I still join the course?", "short"])
    assert vectors.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_padding_does_not_change_a_vector(encoder):
    alone = encoder.encode(["When is the deadline?"])[0]
    batched = encoder.encode(["When is the deadline?", "a much longer sentence " * 10])[0]
    np.testing.assert_allclose(alone, batched, atol=1e-5)


def test_paraphrase_without_shared_words_is_close(encoder):
    # "due" vs "deadline": no keyword overlap, yet clearly related (~0.56 vs ~0.04).
    # Note: MiniLM still scores "When is the homework party?" higher (~0.74), because
    # shared words pull vectors together. That is why the bot fuses vectors with BM25.
    query, paraphrase, unrelated = encoder.encode([
        "When is the homework due?",
        "What is the submission deadline for assignments?",
        "How do I install Docker on Windows?",
    ])
    assert query @ paraphrase > (query @ unrelated) + 0.3
