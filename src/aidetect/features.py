"""Feature extraction orchestrator.

A document is parsed once (:meth:`FeatureExtractor.parse`); afterwards any span
of sentences can be featurised. Sentence-level scores use centred windows of
``window_sentences`` sentences, because a single sentence carries too little
statistical evidence to be scored on its own.
"""
from __future__ import annotations

import math
import os
from typing import Iterable, Sequence

import numpy as np

from . import FEATURE_VERSION
from .config import load_config
from .preprocessing import Document, build_document
from .signals import consistency, semantic, textstats
from .signals.predictability import make_predictability_backend

NAN = float("nan")

# Feature-name prefix -> family. Each family is scored by its own detector.
FAMILY_BY_PREFIX = [
    ("ppl_", "perplexity"),
    ("bur_", "burstiness"),
    ("sty_", "stylometric"),
    ("pun_", "stylometric"),
    ("voc_", "vocabulary"),
    ("syn_", "syntax"),
    ("rep_", "repetition"),
    ("tra_", "repetition"),
    ("sem_", "semantic"),
    ("style_drift", "consistency"),
    ("cp_", "consistency"),
    ("par_", "paragraph"),
]
DETECTOR_FAMILIES = ["perplexity", "burstiness", "stylometric", "vocabulary", "syntax", "semantic", "repetition"]

# Paragraph-structure features are reported but excluded from the classifiers:
# paragraph formatting depends heavily on where a text was collected (forum,
# encyclopedia, scraped essay) and is a known source of dataset artefacts.
DOC_ONLY_PREFIXES = ("style_drift", "cp_", "par_")


def family_of(name: str) -> str:
    for prefix, fam in FAMILY_BY_PREFIX:
        if name.startswith(prefix):
            return fam
    return "other"


class FeatureExtractor:
    def __init__(self, lang: str, profile: str = "lite", config: dict | None = None, device: str = "cpu"):
        self.cfg = config or load_config()
        self.lang = lang
        self.profile = profile
        prof = self.cfg["profiles"][profile]
        self.pred_kind = prof["predictability"]
        self.sem_kind = prof["semantic"]
        self.pred = make_predictability_backend(self.pred_kind, lang, self.cfg, device)
        self.emb = semantic.make_embedder(self.sem_kind, lang, self.cfg, device)
        fc = self.cfg["features"]
        self.window = int(fc["window_sentences"])
        self.style_window = int(fc["style_window"])
        self.style_stride = int(fc["style_stride"])
        self.chunk = int(fc["chunk_tokens"])
        self.mattr_window = int(fc["mattr_window"])
        self.spacy_model = self.cfg["languages"]["spacy_models"].get(lang)

    # ------------------------------------------------------------------ spec
    @property
    def spec(self) -> dict:
        return {
            "feature_version": FEATURE_VERSION,
            "language": self.lang,
            "profile": self.profile,
            "predictability": self.pred_kind,
            "semantic": self.sem_kind,
            "window_sentences": self.window,
            "spacy_model": self.spacy_model,
        }

    # ------------------------------------------------------------------ parsing
    def parse(self, text: str, normalized: bool = False, max_words: int | None = None) -> Document:
        doc = build_document(text, self.lang, max_words=max_words, normalized=normalized, spacy_model=self.spacy_model)
        self.pred.annotate(doc)
        doc.K = self.emb.gram(doc)
        return doc

    # ------------------------------------------------------------------ spans
    def span_features(self, doc: Document, idx: Sequence[int]) -> dict[str, float]:
        R = [doc.sentences[i] for i in idx]
        f: dict[str, float] = {}
        f.update(textstats.predictability(R))
        f.update(textstats.burstiness(R))
        f.update(textstats.stylometry(R, self.lang))
        f.update(textstats.punctuation(R))
        f.update(textstats.vocabulary(R, self.lang, self.chunk, self.mattr_window))
        f.update(textstats.syntax(R, self.chunk))
        f.update(textstats.repetition(R, self.chunk))
        f.update(textstats.transitions(R))
        K = doc.K[np.ix_(list(idx), list(idx))] if doc.K is not None and len(idx) else np.zeros((0, 0))
        f.update(semantic.semantic_features(K))
        return f

    def document_features(self, doc: Document) -> dict[str, float]:
        idx = list(range(len(doc.sentences)))
        f = self.span_features(doc, idx)
        R = doc.sentences
        drift = consistency.style_drift(R, self.style_window, self.style_stride)
        f.update({k: drift[k] for k in ("style_drift_mean", "style_drift_max", "style_drift_std")})
        f["cp_max_stat"] = consistency.cp_feature(R)
        # paragraph-level descriptive statistics
        paras: dict[int, list] = {}
        for r in R:
            paras.setdefault(r.para, []).append(r)
        plen = [sum(r.c("n_words") for r in p) for p in paras.values()]
        f["par_len_cv"] = textstats._cv(plen)
        pmeans = [float(np.concatenate([r.surprisal for r in p]).mean()) for p in paras.values()
                  if sum(len(r.surprisal) for r in p) >= 5]
        f["par_ppl_std"] = textstats._std(pmeans)
        f["par_sem_adjacent"] = semantic.paragraph_similarity(doc.K, [r.para for r in R]) if doc.K is not None and len(R) else NAN
        firsts = [p[0] for p in paras.values()]
        f["par_transition_initial"] = (sum(1 for r in firsts if r.opening == "transition") / len(firsts)
                                       if len(firsts) >= 2 else NAN)
        return f

    def window_indices(self, n: int, center: int) -> list[int]:
        half = self.window // 2
        lo = max(0, center - half)
        hi = min(n, lo + self.window)
        lo = max(0, hi - self.window)
        return list(range(lo, hi))

    def window_features(self, doc: Document, centers: Iterable[int] | None = None) -> list[tuple[int, list[int], dict]]:
        n = len(doc.sentences)
        centers = range(n) if centers is None else centers
        out = []
        for c in centers:
            idx = self.window_indices(n, c)
            out.append((c, idx, self.span_features(doc, idx)))
        return out


def model_feature_names(feature_dict: dict, level: str) -> list[str]:
    names = []
    for k in feature_dict:
        fam = family_of(k)
        if fam == "paragraph":
            continue
        if level == "window" and k.startswith(DOC_ONLY_PREFIXES):
            continue
        names.append(k)
    return names


def to_matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    X = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(names):
            v = r.get(k, NAN)
            X[i, j] = v if v is not None and not (isinstance(v, float) and math.isinf(v)) else NAN
    return X


# --------------------------------------------------------------------------- corpus featurisation
_WORKER: dict = {}


def _init_worker(lang, profile, config):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER["fx"] = FeatureExtractor(lang, profile, config)


def _featurize_one(args):
    sample, windows_per_doc, seed, max_words = args
    fx: FeatureExtractor = _WORKER["fx"]
    try:
        doc = fx.parse(sample["text"], normalized=True, max_words=max_words)
    except Exception as exc:  # noqa: BLE001 - corpus building must not die on one bad sample
        return {"id": sample["id"], "error": repr(exc)}
    n = len(doc.sentences)
    if n == 0:
        return {"id": sample["id"], "error": "no sentences"}
    sent_labels = sentence_labels(doc, sample)
    out = {"id": sample["id"], "n_sentences": n, "n_words": doc.n_words, "doc": fx.document_features(doc),
           "sentence_labels": sent_labels.tolist(),
           "sentence_words": [int(s.c("n_words")) for s in doc.sentences], "windows": None}
    if windows_per_doc:
        rng = np.random.default_rng(seed)
        centers = list(range(n))
        if len(centers) > windows_per_doc:
            centers = sorted(rng.choice(centers, size=windows_per_doc, replace=False).tolist())
        rows = fx.window_features(doc, centers)
        names = list(rows[0][2].keys())
        out["windows"] = {
            "names": names,
            "centers": [c for c, _, _ in rows],
            "ai_fraction": [float(np.mean(sent_labels[idx])) for _, idx, _ in rows],
            "center_label": [float(sent_labels[c]) for c, _, _ in rows],
            "X": np.array([[f[k] for k in names] for _, _, f in rows], dtype=np.float32),
        }
    return out


def sentence_labels(doc: Document, sample: dict) -> np.ndarray:
    """Per-sentence AI labels: from ``segments`` (hybrid docs) or the document label."""
    segs = sample.get("segments")
    if not segs:
        return np.full(len(doc.sentences), float(sample.get("label", 0)))
    labels = np.zeros(len(doc.sentences))
    for i, s in enumerate(doc.sentences):
        mid = (s.start + s.end) / 2
        for a, b, lab in segs:
            if a <= mid < b:
                labels[i] = lab
                break
    return labels


def featurize_corpus(samples: list[dict], lang: str, profile: str = "lite", config: dict | None = None,
                     n_jobs: int = 1, windows_per_doc: int = 0, seed: int = 0, max_words: int | None = None,
                     progress: bool = True) -> list[dict]:
    """Featurise many samples (multiprocessing). Samples must contain normalised text."""
    config = config or load_config()
    args = [(s, windows_per_doc, seed + i, max_words) for i, s in enumerate(samples)]
    results = []
    if n_jobs <= 1:
        _init_worker(lang, profile, config)
        for i, a in enumerate(args):
            results.append(_featurize_one(a))
            if progress and (i + 1) % 500 == 0:
                print(f"  featurised {i + 1}/{len(args)}", flush=True)
        return results
    import multiprocessing as mp

    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(n_jobs, initializer=_init_worker, initargs=(lang, profile, config)) as pool:
        for i, r in enumerate(pool.imap(_featurize_one, args, chunksize=8)):
            results.append(r)
            if progress and (i + 1) % 500 == 0:
                print(f"  featurised {i + 1}/{len(args)}", flush=True)
    return results
