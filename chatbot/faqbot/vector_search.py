"""In-memory vector index: exact cosine-similarity search with numpy.

1,427 documents × 384 dimensions is ~2 MB, so brute force takes well under a
millisecond and needs no approximate-nearest-neighbour library or vector database.
"""

import numpy as np


class VectorIndex:
    def __init__(self, keyword_fields: list[str]):
        self.keyword_fields = keyword_fields

    def fit(self, vectors: np.ndarray, docs: list[dict]) -> "VectorIndex":
        if len(vectors) != len(docs):
            raise ValueError(f"{len(vectors)} vectors for {len(docs)} documents")
        # Rows are L2-normalised, so a dot product is the cosine similarity.
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.docs = docs
        self.keyword_values = {
            f: np.array([d.get(f) for d in docs], dtype=object) for f in self.keyword_fields
        }
        return self

    def search(
        self, query_vector: np.ndarray, filter_dict: dict | None = None, num_results: int = 5
    ) -> list[dict]:
        scores = self.vectors @ query_vector
        for field, value in (filter_dict or {}).items():
            scores = np.where(self.keyword_values[field] == value, scores, -np.inf)

        top = np.argsort(-scores)[:num_results]
        return [self.docs[i] for i in top if np.isfinite(scores[i])]
