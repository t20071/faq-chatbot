"""Embed every FAQ record with sentence-transformers (all-MiniLM-L6-v2).

Writes:
  data/embeddings.npy    float32 matrix, one L2-normalised 384-dim row per record
  data/embeddings.json   model, revision, text template and the record ids (row order)

Embeddings are computed by sentence-transformers' ONNX backend from the same quantised
model file the app uses for queries (models/), and the script checks the two encoders
agree. Run from the project root after build_data.py:

    uv run python chatbot/scripts/build_embeddings.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faqbot.config import (  # noqa: E402
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_TEXT_TEMPLATE,
    EMBEDDINGS_META_PATH,
    EMBEDDINGS_PATH,
    ONNX_REPO_FILE,
)
from faqbot.data import load_documents  # noqa: E402
from faqbot.embeddings import MiniLMEncoder  # noqa: E402
from faqbot.model_files import ensure_model_files  # noqa: E402

PARITY_THRESHOLD = 0.999  # minimum cosine similarity between the two encoders
PARITY_QUERIES = [
    "Can I still join the course?",
    "when is the capstone due?",
    "is groq free?",
    "docker: permission denied while trying to connect to the daemon socket",
]


def main():
    documents = load_documents()
    texts = [EMBEDDING_TEXT_TEMPLATE.format(**d) for d in documents]

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        revision=EMBEDDING_MODEL_REVISION,
        backend="onnx",
        model_kwargs={"file_name": ONNX_REPO_FILE},
    )
    model.max_seq_length = EMBEDDING_MAX_TOKENS
    # batch_size=1: the quantised model's output depends on what else is in the batch,
    # so each text is embedded alone, exactly like queries are in the app.
    vectors = model.encode(
        texts, batch_size=1, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)

    np.save(EMBEDDINGS_PATH, vectors)
    EMBEDDINGS_META_PATH.write_text(json.dumps({
        "model": EMBEDDING_MODEL,
        "revision": EMBEDDING_MODEL_REVISION,
        "onnx_file": ONNX_REPO_FILE,
        "text_template": EMBEDDING_TEXT_TEMPLATE,
        "dimensions": int(vectors.shape[1]),
        "ids": [d["id"] for d in documents],
    }), encoding="utf-8")
    print(f"Saved {vectors.shape} embeddings to {EMBEDDINGS_PATH}")

    # The app embeds queries with ONNX Runtime; make sure it matches sentence-transformers.
    encoder = MiniLMEncoder(model_dir=ensure_model_files())
    sample = PARITY_QUERIES + texts[:: max(1, len(texts) // 100)]
    reference = model.encode(sample, batch_size=1, normalize_embeddings=True)
    similarity = (reference * encoder.encode(sample)).sum(axis=1)
    print(f"ONNX vs sentence-transformers cosine: min={similarity.min():.5f} "
          f"mean={similarity.mean():.5f} over {len(sample)} texts")
    if similarity.min() < PARITY_THRESHOLD:
        sys.exit(f"Encoders disagree (min cosine < {PARITY_THRESHOLD}); vectors would not match")


if __name__ == "__main__":
    main()
