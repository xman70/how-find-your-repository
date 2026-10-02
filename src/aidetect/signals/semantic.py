"""Semantic consistency features from sentence embeddings.

Backends
``lexical``               Hashed content-word unigrams + character 3-5-grams
                          (morphology-tolerant, works for Greek inflection). No
                          download needed; measures lexical rather than deep
                          semantic relatedness.
``sentence_transformer``  Multilingual sentence-transformer embeddings (optional).

All features are computed from the cosine-similarity (Gram) matrix ``K`` of the
L2-normalised sentence vectors, so any span can be analysed by indexing ``K``.
"""
from __future__ import annotations

import math

import numpy as np

from ..lexicons import LEXICONS
from ..preprocessing import Document

NAN = float("nan")


class LexicalEmbedder:
    name = "lexical"

    def __init__(self, lang: str, n_features: int = 2 ** 18):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.function = LEXICONS[lang]["function"]
        self.word_vec = HashingVectorizer(analyzer=lambda ws: ws, n_features=n_features, alternate_sign=False, norm="l2")
        self.char_vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=n_features,
                                          alternate_sign=False, norm="l2", lowercase=False)

    def gram(self, doc: Document) -> np.ndarray:
        from sklearn.preprocessing import normalize

        contents = [[w for w in s.words if w not in self.function] for s in doc.sentences]
        if not contents:
            return np.zeros((0, 0))
        W = self.word_vec.transform(contents)
        C = self.char_vec.transform([" ".join(c) for c in contents])
        E = normalize(W + C, norm="l2")
        K = (E @ E.T).toarray()
        return K


class SentenceTransformerEmbedder:
    name = "sentence_transformer"

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed (requirements-optional.txt).") from exc
        self.model = SentenceTransformer(model_name, device=device)

    def gram(self, doc: Document) -> np.ndarray:
        texts = [s.text for s in doc.sentences]
        if not texts:
            return np.zeros((0, 0))
        E = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(E @ E.T, dtype=np.float64)


def make_embedder(kind: str, lang: str, config: dict | None = None, device: str = "cpu"):
    if kind == "lexical":
        return LexicalEmbedder(lang)
    if kind == "sentence_transformer":
        name = (config or {}).get("profiles", {}).get("full", {}).get("sentence_transformer")
        return SentenceTransformerEmbedder(name, device=device)
    raise ValueError(f"Unknown semantic backend: {kind}")


def centroid_sims(K: np.ndarray) -> np.ndarray:
    tot = K.mean()
    if tot <= 1e-12:
        return np.zeros(K.shape[0])
    return K.mean(axis=1) / math.sqrt(tot)


def semantic_features(K: np.ndarray) -> dict[str, float]:
    """Features of a span's similarity matrix (n x n)."""
    n = K.shape[0]
    f = {k: NAN for k in ("sem_adj_mean", "sem_adj_std", "sem_adj_min", "sem_centroid_mean", "sem_centroid_std",
                          "sem_redundancy", "sem_local_gain", "sem_topic_jump_rate", "sem_compression")}
    if n < 2:
        return f
    adj = np.array([K[i, i + 1] for i in range(n - 1)])
    f["sem_adj_mean"] = float(adj.mean())
    f["sem_adj_min"] = float(adj.min())
    if n >= 3:
        f["sem_adj_std"] = float(adj.std())
        cs = centroid_sims(K)
        f["sem_centroid_mean"] = float(cs.mean())
        f["sem_centroid_std"] = float(cs.std())
    if n >= 4:
        idx = np.arange(n)
        far = np.abs(idx[:, None] - idx[None, :]) >= 2
        far_vals = K[far]
        nonadj_mean = float(far_vals.mean())
        Kf = np.where(far, K, -np.inf)
        f["sem_redundancy"] = float(np.max(Kf, axis=1).mean())
        f["sem_local_gain"] = float(adj.mean() - nonadj_mean)
        f["sem_topic_jump_rate"] = float(np.mean(adj < nonadj_mean))
        H = np.eye(n) - 1.0 / n
        ev = np.linalg.eigvalsh(H @ K @ H)
        ev = np.clip(ev, 0, None)
        if ev.sum() > 1e-12:
            f["sem_compression"] = float(ev.max() / ev.sum())
    return f


def paragraph_similarity(K: np.ndarray, para_of_sentence: list[int]) -> float:
    paras = sorted(set(para_of_sentence))
    if len(paras) < 2:
        return NAN
    groups = [np.array([i for i, p in enumerate(para_of_sentence) if p == q]) for q in paras]
    sims = []
    for a, b in zip(groups, groups[1:]):
        num = K[np.ix_(a, b)].mean()
        den = math.sqrt(max(K[np.ix_(a, a)].mean(), 1e-12) * max(K[np.ix_(b, b)].mean(), 1e-12))
        sims.append(num / den)
    return float(np.mean(sims))
