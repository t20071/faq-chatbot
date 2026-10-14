"""Small pure-Python BM25 index.

minsearch pulls in numpy/pandas/scikit-learn, which bloats a Vercel function
bundle. This index mirrors its interface (text fields, keyword filters,
per-field boosts) with zero dependencies.
"""

import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "so", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
}


def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


class Index:
    def __init__(self, text_fields, keyword_fields, k1=1.5, b=0.75):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.k1 = k1
        self.b = b
        self.docs = []

    def fit(self, docs):
        self.docs = docs
        n = len(docs)
        # field -> list of Counter(term -> tf), one per doc
        self.tfs = {f: [] for f in self.text_fields}
        self.lengths = {f: [] for f in self.text_fields}
        # field -> term -> list of doc ids containing it
        self.postings = {f: defaultdict(list) for f in self.text_fields}

        for i, doc in enumerate(docs):
            for f in self.text_fields:
                tokens = tokenize(doc.get(f) or "")
                tf = Counter(tokens)
                self.tfs[f].append(tf)
                self.lengths[f].append(len(tokens))
                for term in tf:
                    self.postings[f][term].append(i)

        self.avg_len = {
            f: (sum(self.lengths[f]) / n if n else 0.0) or 1.0
            for f in self.text_fields
        }
        self.idf = {
            f: {
                term: math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
                for term, ids in self.postings[f].items()
            }
            for f in self.text_fields
        }
        return self

    def search(self, query, filter_dict=None, boost_dict=None, num_results=5):
        filter_dict = filter_dict or {}
        boost_dict = boost_dict or {}
        terms = set(tokenize(query))
        scores = defaultdict(float)

        for f in self.text_fields:
            boost = boost_dict.get(f, 1.0)
            if boost == 0:
                continue
            avg_len = self.avg_len[f]
            for term in terms:
                idf = self.idf[f].get(term)
                if idf is None:
                    continue
                for i in self.postings[f][term]:
                    tf = self.tfs[f][i][term]
                    norm = 1 - self.b + self.b * self.lengths[f][i] / avg_len
                    scores[i] += boost * idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)

        def matches(doc):
            return all(doc.get(k) == v for k, v in filter_dict.items())

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        results = []
        for i, _score in ranked:
            if matches(self.docs[i]):
                results.append(self.docs[i])
                if len(results) == num_results:
                    break
        return results
