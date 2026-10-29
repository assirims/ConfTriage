"""Lexical retrieval baselines: TF-IDF cosine and Okapi BM25 over the ESCO occupation documents."""
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]*\b"


def clean(t):
    return re.sub(r"\s+", " ", (t or "")).strip()


class TfidfIndex:
    def __init__(self, docs, ngram=(1, 2), min_df=1):
        self.vec = TfidfVectorizer(token_pattern=TOKEN, lowercase=True, ngram_range=ngram, min_df=min_df,
                                   sublinear_tf=True, stop_words="english")
        self.D = self.vec.fit_transform([clean(d) for d in docs])

    def scores(self, queries):
        Q = self.vec.transform([clean(q) for q in queries])
        return (Q @ self.D.T).toarray()


class BM25Index:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.vec = CountVectorizer(token_pattern=TOKEN, lowercase=True, stop_words="english")
        tf = self.vec.fit_transform([clean(d) for d in docs]).tocsc().astype(np.float64)
        N = tf.shape[0]
        dl = np.asarray(tf.sum(1)).ravel()
        avg = dl.mean()
        df = np.asarray((tf > 0).sum(0)).ravel()
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
        w = tf.data * (k1 + 1) / denom
        self.W = sparse.csr_matrix((w * self.idf[tf.col], (tf.row, tf.col)), shape=tf.shape)

    def scores(self, queries):
        Q = self.vec.transform([clean(q) for q in queries])
        Q.data[:] = 1.0                       # binary query term weights
        return (Q @ self.W.T).toarray()
