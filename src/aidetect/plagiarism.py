"""Similarity / plagiarism analysis - completely separate from AI detection.

Answers "does this text overlap with known documents?", never "was it written by
an AI?". Compares the input against a *local* reference corpus that the user
builds (stored in SQLite); nothing is sent to the internet. Internet search is
not included: connecting to a web index would upload the text and is therefore
left out of the local-first design.

Measures
  exact phrase matches   maximal shared word sequences of >= ``min_match_words`` words
  n-gram containment     share of the input's word 3-/5-grams present in a reference
  document similarity    cosine similarity of TF-IDF-weighted word vectors
  semantic similarity    best-matching reference sentence per input sentence
                         (lexical embedding backend; paraphrase-level matching is limited)
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from difflib import SequenceMatcher

import numpy as np

from .preprocessing import normalize_text

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _hash(gram: tuple[str, ...]) -> int:
    return zlib.crc32(" ".join(gram).encode("utf-8")) & 0x7FFFFFFF


def winnow(toks: list[str], k: int = 5, w: int = 4) -> set[int]:
    """Winnowing fingerprints (Schleimer et al., 2003) of word k-grams."""
    if len(toks) < k:
        return {_hash(tuple(toks))} if toks else set()
    hs = [_hash(tuple(toks[i:i + k])) for i in range(len(toks) - k + 1)]
    if len(hs) <= w:
        return {min(hs)}
    return {min(hs[i:i + w]) for i in range(len(hs) - w + 1)}


def ngram_containment(a: list[str], b: list[str], n: int) -> float:
    if len(a) < n or len(b) < n:
        return 0.0
    ga = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1))
    gb = set(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
    tot = sum(ga.values())
    return sum(v for g, v in ga.items() if g in gb) / tot


def exact_matches(a: list[str], b: list[str], min_words: int = 8) -> list[tuple[int, int, int]]:
    """(start_in_a, start_in_b, length) for shared word runs of at least min_words."""
    sm = SequenceMatcher(None, a, b, autojunk=False)
    return [(m.a, m.b, m.size) for m in sm.get_matching_blocks() if m.size >= min_words]


def cosine_tfidf(a: list[str], b: list[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    vocab = set(ca) | set(cb)
    idf = {t: math.log(3 / (1 + (t in ca) + (t in cb))) + 1 for t in vocab}
    va = {t: ca[t] * idf[t] for t in ca}
    vb = {t: cb[t] * idf[t] for t in cb}
    num = sum(va[t] * vb.get(t, 0.0) for t in va)
    den = math.sqrt(sum(v * v for v in va.values())) * math.sqrt(sum(v * v for v in vb.values()))
    return num / den if den else 0.0


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 5]


def _sentence_vectors(sents: list[str]):
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 18, alternate_sign=False, norm="l2")
    return hv.transform(sents)


def semantic_matches(input_text: str, ref_text: str, threshold: float = 0.8) -> list[dict]:
    a, b = _sentences(input_text), _sentences(ref_text)
    if not a or not b:
        return []
    S = (_sentence_vectors(a) @ _sentence_vectors(b).T).toarray()
    out = []
    for i in range(len(a)):
        j = int(np.argmax(S[i]))
        if S[i, j] >= threshold:
            out.append({"input_sentence": a[i], "reference_sentence": b[j], "similarity": float(S[i, j])})
    return out


class SimilarityChecker:
    def __init__(self, store, cfg: dict | None = None):
        self.store = store
        c = (cfg or {}).get("similarity", {})
        self.k = int(c.get("shingle_size", 5))
        self.w = int(c.get("winnow_window", 4))
        self.min_words = int(c.get("min_match_words", 8))
        self.sem_thr = float(c.get("semantic_threshold", 0.8))

    def add_reference(self, title: str, text: str) -> int | None:
        text = normalize_text(text)
        return self.store.add_reference(title, text, winnow(tokens(text), self.k, self.w))

    def check(self, text: str, max_sources: int = 10) -> dict:
        text = normalize_text(text)
        toks = tokens(text)
        fps = winnow(toks, self.k, self.w)
        cands = self.store.candidate_references(fps, limit=max_sources * 2)
        sources = []
        covered = np.zeros(len(toks), dtype=bool)
        for cnd in cands:
            rt = tokens(cnd["text"])
            matches = exact_matches(toks, rt, self.min_words)
            for a0, _, size in matches:
                covered[a0:a0 + size] = True
            sem = semantic_matches(text, cnd["text"], self.sem_thr)
            src = {
                "reference_id": cnd["id"], "title": cnd["title"],
                "exact_match_words": int(sum(m[2] for m in matches)),
                "exact_matches": [{"input_start_word": a0, "length_words": size,
                                   "text": " ".join(toks[a0:a0 + size])} for a0, _, size in matches[:20]],
                "containment_3gram": ngram_containment(toks, rt, 3),
                "containment_5gram": ngram_containment(toks, rt, 5),
                "document_cosine": cosine_tfidf(toks, rt),
                "semantic_sentence_matches": sem[:20],
            }
            src["score"] = max(src["containment_5gram"], src["exact_match_words"] / max(len(toks), 1))
            sources.append(src)
        sources.sort(key=lambda s: -s["score"])
        n_refs = len(self.store.list_references())
        return {
            "kind": "similarity",
            "reference_corpus_size": n_refs,
            "overall_exact_overlap": float(covered.mean()) if len(toks) else 0.0,
            "sources": sources[:max_sources],
            "note": ("Similarity analysis compares the text with the local reference corpus only "
                     f"({n_refs} documents). It is independent of the AI-likelihood estimate: overlap is not evidence "
                     "of AI generation, and low overlap is not evidence of original authorship."),
        }
