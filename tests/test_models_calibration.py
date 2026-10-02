import numpy as np
import pytest

from aidetect.calibration import (IsotonicCalibrator, PlattCalibrator, balanced_weights, brier, compare_calibrators,
                                  ece, reliability)
from aidetect.metrics import binary_metrics, bootstrap_ci, rate_ci
from aidetect.models import Ensemble, percentile_in


def _miscalibrated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    true_p = np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    raw = true_p ** 3  # systematically under-confident for the positive class
    return raw, y


def test_calibrators_reduce_ece():
    raw, y = _miscalibrated()
    w = balanced_weights(y)
    base = ece(y, raw, 10, w)
    for cal in (PlattCalibrator(), IsotonicCalibrator()):
        p = cal.fit(raw[:2000], y[:2000], balanced_weights(y[:2000])).transform(raw[2000:])
        assert ece(y[2000:], p, 10, balanced_weights(y[2000:])) < base
        assert p.min() >= 0.005 and p.max() <= 0.995  # never claims certainty


def test_compare_calibrators_reports_both():
    raw, y = _miscalibrated()
    rep = compare_calibrators(raw, y)
    assert set(rep["results"]) == {"none", "platt", "isotonic"}
    assert rep["selected"] in ("platt", "isotonic")
    assert len(reliability(y, raw)["bins"]) == 10
    assert 0 <= brier(y, raw) <= 1


def test_metrics_and_intervals():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.7, 0.8, 0.4, 0.9, 0.2])
    m = binary_metrics(y, p)
    assert m["fpr"] == pytest.approx(1 / 3) and m["fnr"] == pytest.approx(1 / 3)
    assert 0 <= m["roc_auc"] <= 1
    ci = bootstrap_ci(np.tile(y, 20), np.tile(p, 20), n_boot=50)
    assert ci["accuracy"][0] <= m["accuracy"] <= ci["accuracy"][1]
    lo, hi = rate_ci(5, 100)
    assert lo < 0.05 < hi
    assert rate_ci(0, 0) is None


def test_ensemble_learns_and_weights_are_valid():
    rng = np.random.default_rng(1)
    n = 600
    y = rng.integers(0, 2, n)
    names = ["ppl_token_mean", "sty_sent_len_mean", "voc_mattr", "syn_depth_mean", "rep_bigram", "sem_adj_mean",
             "bur_sent_len_cv"]
    X = rng.normal(0, 1, (n, len(names)))
    X[:, 0] += 1.5 * y
    X[:, 2] -= 1.0 * y
    X[rng.random((n, len(names))) < 0.05] = np.nan  # missing values are imputed
    groups = np.arange(n) // 3
    ens = Ensemble("document", names, seed=0, folds=3).fit(X, y, groups, verbose=False)
    ens.calibrate(X, y)
    p = ens.predict_proba(X)
    assert ((p >= 0.5) == y).mean() > 0.75
    w = ens.weights
    assert all(v >= 0 for v in w.values()) and sum(w.values()) == pytest.approx(1.0)
    det = ens.predict_detail(X[:5])["detectors"]
    assert {"prob", "std", "confidence", "coverage"} <= set(next(iter(det.values())))
    assert "ppl_token_mean" in ens.population and "human" in ens.population["ppl_token_mean"]


def test_percentile_helper():
    q = list(np.linspace(0, 10, 21))
    assert percentile_in(q, 5.0) == pytest.approx(50.0)
    assert percentile_in(q, -1) == 0.0
