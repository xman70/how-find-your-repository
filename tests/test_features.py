import math

import numpy as np

from aidetect.features import FeatureExtractor, family_of, model_feature_names
from aidetect.preprocessing import build_document
from aidetect.signals import consistency, semantic, textstats

from conftest import AIISH, HUMANISH


def test_document_and_window_features(cfg):
    fx = FeatureExtractor("en", "lite", cfg)
    doc = fx.parse(HUMANISH + "\n\n" + AIISH)
    f = fx.document_features(doc)
    assert len(f) > 100
    assert all(not (isinstance(v, float) and math.isinf(v)) for v in f.values())
    names = model_feature_names(f, "document")
    assert not any(n.startswith("par_") for n in names)  # paragraph features never enter models
    fams = {family_of(n) for n in names}
    assert {"perplexity", "burstiness", "stylometric", "vocabulary", "syntax", "semantic", "repetition"} <= fams
    rows = fx.window_features(doc)
    assert len(rows) == len(doc.sentences)
    assert all(len(idx) <= fx.window for _, idx, _ in rows)
    assert not any(k.startswith("style_drift") for k in rows[0][2])


def test_feature_spec_records_backends(cfg):
    spec = FeatureExtractor("en", "lite", cfg).spec
    assert spec["predictability"] == "unigram" and spec["semantic"] == "lexical"
    assert spec["feature_version"]


def test_lexical_measures():
    words = ("the cat sat on the mat and the dog sat on the log " * 10).split()
    assert 0 < textstats.mattr(words, 50) < 1
    assert textstats.mtld(words[:30]) != textstats.mtld(words[:30]) or True  # short -> NaN allowed
    assert math.isnan(textstats.mtld(words[:30]))
    assert textstats.yule_k(words) > 0
    assert textstats.ngram_repetition(words, 3, 150) > 0.5
    assert textstats.burst_index([5, 5, 5, 5]) == -1.0
    assert textstats.entropy([1, 1, 1, 1], 4) == 1.0


def test_js_divergence_and_drift(cfg):
    a = np.array([10.0, 0, 0])
    assert consistency.js_divergence(a, a) < 1e-9
    assert consistency.js_divergence(a, np.array([0, 0, 10.0])) > 0.5
    doc = build_document(HUMANISH + " " + AIISH, "en")
    d = consistency.style_drift(doc.sentences, 5, 2)
    assert d["series"] and not math.isnan(d["style_drift_mean"])


def test_changepoint_detects_planted_shift():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0, 0.3, (12, 4)), rng.normal(3, 0.3, (12, 4))])
    T = consistency.boundary_stats(Z, window=4, min_segment=3)
    assert int(np.nanargmax(T)) == 12


def test_changepoint_end_to_end(cfg):
    fx = FeatureExtractor("en", "lite", cfg)
    doc = fx.parse(HUMANISH + " " + AIISH)
    res = consistency.detect_changepoints(doc.sentences, doc.K, permutations=50)
    assert res["p_value"] is not None
    for c in res["changepoints"]:
        assert c["label"] == "Potential authorship/style transition"


def test_semantic_features_on_gram_matrix():
    K = np.eye(5)
    K[0, 1] = K[1, 0] = 0.8
    f = semantic.semantic_features(K)
    assert 0 <= f["sem_adj_mean"] <= 1 and not math.isnan(f["sem_compression"])
    assert math.isnan(semantic.semantic_features(np.eye(1))["sem_adj_mean"])
