"""Model, calibration, conformal, ensemble and reproducibility tests."""
import numpy as np
import pandas as pd
import pytest

from vusa_quant_ai.labeling.labels import build_labels
from vusa_quant_ai.models.calibration.calibration import ConformalRegressor, ProbabilityCalibrator, rolling_conformal_backtest
from vusa_quant_ai.models.registry import ALL_MODELS
from vusa_quant_ai.validation.metrics import ece


@pytest.fixture(scope="module")
def xy(features, bundle):
    X = features.frame[features.model_columns()[:30]].iloc[300:]
    lab = build_labels(bundle.target, [5])[5].reindex(X.index)
    ok = lab["fwd_ret"].notna()
    return X[ok], lab[ok]


@pytest.mark.parametrize("name", ["naive_drift", "elastic_net", "random_forest", "extra_trees", "hist_gb", "xgboost",
                                  "lightgbm", "catboost"])
def test_ml_models_fit_predict(name, xy):
    cls = ALL_MODELS[name]
    if not cls.info.available:
        pytest.skip(cls.info.unavailable_reason)
    X, lab = xy
    tr, te = X.iloc[:1200], X.iloc[1200:1400]
    m = cls(5, seed=1).fit(tr, lab["fwd_ret"].iloc[:1200], lab["direction"].iloc[:1200])
    p = m.predict(te)
    assert list(p.columns) == ["ret", "p_up"]
    assert p["p_up"].between(0, 1).all()
    assert np.isfinite(p["ret"]).all()


@pytest.mark.parametrize("name", ["arima", "ets", "state_space"])
def test_statistical_models_are_causal(name, bundle, xy):
    """Forecast at origin t must not change when data after t is appended."""
    X, lab = xy
    close = bundle.target["close"]
    m = ALL_MODELS[name](5)
    m.set_context(None, close)
    tr = X.iloc[:1500]
    m.fit(tr, lab["fwd_ret"].iloc[:1500], lab["direction"].iloc[:1500])
    origin = X.index[1600]
    p_short = m.predict(X.loc[[origin]])
    m.set_context(None, close.loc[: X.index[1800]])
    p_long = m.predict(X.loc[[origin]])
    assert p_short["ret"].iloc[0] == pytest.approx(p_long["ret"].iloc[0], rel=1e-6, abs=1e-9)


def test_deep_learning_model_or_reported_unavailable(xy):
    cls = ALL_MODELS["gru"]
    if not cls.info.available:
        assert "PyTorch" in cls.info.unavailable_reason
        return
    X, lab = xy
    m = cls(5, seed=0, epochs=2, lookback=10)
    m.set_context(X, None)
    m.fit(X.iloc[:800], lab["fwd_ret"].iloc[:800], lab["direction"].iloc[:800])
    p = m.predict(X.iloc[800:850])
    assert len(p) == 50 and p["p_up"].between(0, 1).all()


def test_isotonic_calibration_improves_ece():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.2, 0.8, 4000)
    y = (rng.random(4000) < true_p).astype(float)
    raw = np.clip(0.5 + 1.8 * (true_p - 0.5), 0.01, 0.99)  # over-confident
    cal = ProbabilityCalibrator().fit(raw[:3000], y[:3000])
    assert cal.chosen_ in ("isotonic", "platt")
    assert ece(y[3000:], cal.transform(raw[3000:])) < ece(y[3000:], raw[3000:])


def test_conformal_coverage_near_nominal():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.standard_t(4, 3000) * 0.02)
    pred = pd.Series(np.zeros(3000))
    bt = rolling_conformal_backtest(y, pred, alpha=0.2, window=500, min_n=200)
    m = bt.dropna()
    cov = ((m["y"] >= m["lo"]) & (m["y"] <= m["hi"])).mean()
    assert 0.75 < cov < 0.85
    cr = ConformalRegressor(0.2).fit(y.values, pred.values)
    lo, hi = cr.interval(0.0)
    assert lo < 0 < hi
    assert 0 <= cr.prob_above(0.0, 0.05) <= 1


def test_forecast_service_end_to_end_and_reproducible(features, bundle, regimes, settings):
    from vusa_quant_ai.models.forecast_service import ForecastService

    svc = ForecastService(settings, hpo_trials=0)
    models = ["naive_drift", "elastic_net", "hist_gb"]
    r1 = svc.run(features, bundle.target, regimes, [5], models, register=False)
    r2 = svc.run(features, bundle.target, regimes, [5], models, register=False)
    h1, h2 = r1.horizons[5], r2.horizons[5]
    assert r1.audit.passed
    assert np.isfinite(h1.expected_return) and 0 < h1.p_up < 1
    assert h1.interval[0] < h1.expected_return < h1.interval[1]
    assert h1.expected_return == pytest.approx(h2.expected_return)
    assert h1.p_up == pytest.approx(h2.p_up)
    assert set(h1.ensemble_comparison) == {"best_single", "equal_weight", "performance_weighted", "stacked", "bayesian"}
    assert h1.n_oos > 300
    assert "naive_drift" in h1.model_scores


def test_unknown_or_missing_model_is_reported(features, bundle, regimes, settings):
    from vusa_quant_ai.models.forecast_service import ForecastService

    ok, skipped = ForecastService(settings).available_models(["not_a_model", "hist_gb"])
    assert "not_a_model" in skipped and "naive_drift" in ok


def test_hpo_nested_runs(xy):
    from vusa_quant_ai.models.hpo import tune
    from vusa_quant_ai.models.tree.models import LGBMForecaster

    if not LGBMForecaster.info.available:
        pytest.skip("lightgbm missing")
    X, lab = xy
    r = tune("lightgbm", X.iloc[:1500], lab["fwd_ret"].iloc[:1500], lab["direction"].iloc[:1500],
             lab["label_end"].iloc[:1500], 5, n_trials=2)
    assert "num_leaves" in r["params"]
