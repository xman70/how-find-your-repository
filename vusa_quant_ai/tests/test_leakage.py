"""Leakage tests: causality of features, purging, audit failures stop backtests."""
import numpy as np
import pandas as pd
import pytest

from vusa_quant_ai.features.technical.indicators import technical_features
from vusa_quant_ai.labeling.labels import build_labels, forward_return, triple_barrier
from vusa_quant_ai.validation.leakage import LeakageError, check_feature_causality, run_audit
from vusa_quant_ai.validation.splits import purged_kfold, walk_forward


def test_technical_features_are_causal(bundle):
    data = bundle.target[["open", "high", "low", "close", "volume"]].iloc[-1200:]
    chk = check_feature_causality(technical_features, data, [600, 900, 1100])
    assert chk.status == "PASS", chk.detail


def test_causality_check_catches_centered_window(bundle):
    data = bundle.target[["open", "high", "low", "close", "volume"]].iloc[-800:]

    def leaky(d):
        return pd.DataFrame({"centered_ma": d["close"].rolling(11, center=True).mean()}, index=d.index)

    chk = check_feature_causality(leaky, data, [400, 600])
    assert chk.status == "FAIL"


def test_forward_return_uses_next_open():
    idx = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame({"open": [10, 11, 12, 13, 14, 15.0], "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5]}, index=idx)
    fr = forward_return(df, 2)
    assert fr.iloc[0] == pytest.approx(12.5 / 11 - 1)  # entry next open (11), exit close t+2


def test_triple_barrier_labels():
    c = pd.Series([100, 101, 103, 106, 106, 106, 106, 100, 97, 94, 94, 94.0], index=pd.bdate_range("2024-01-01", periods=12))
    vol = pd.Series(0.01, index=c.index)
    tb = triple_barrier(c, 3, vol, 1.5, 1.5, min_barrier=0.02)
    assert tb["tb_label"].iloc[0] == 1  # hits +2.6% quickly
    assert tb["tb_label"].iloc[6] == -1  # falls 6%
    assert tb["tb_label"].iloc[3] == 0  # flat -> timeout


def test_purging_removes_overlapping_labels(bundle):
    idx = bundle.target.index[:1500]
    lab = build_labels(bundle.target.iloc[:1500], [20])[20]
    splits = list(walk_forward(idx, lab["label_end"], 3, 600, embargo=5))
    assert splits
    for sp in splits:
        t0 = idx[sp.test[0]]
        before = idx[sp.train] < t0
        assert (lab["label_end"].iloc[sp.train][before] < t0).all()
    for sp in purged_kfold(idx, lab["label_end"], 4, embargo=10):
        t0, t1 = idx[sp.test[0]], lab["label_end"].iloc[sp.test].max()
        starts, ends = idx[sp.train], lab["label_end"].iloc[sp.train]
        assert not ((starts <= t1) & (ends >= t0)).any()


def test_audit_fails_on_injected_future_information(bundle, features):
    lab = build_labels(bundle.target, [5])[5]
    X = features.frame[features.model_columns()].copy()
    X["leak"] = bundle.target["close"].shift(-5) / bundle.target["close"] - 1  # future price!
    audit = run_audit(X.iloc[:-10], lab["fwd_ret"].iloc[:-10], bundle.target["close"], X.index[-11])
    assert not audit.passed
    names = {c.name for c in audit.checks if c.status == "FAIL"}
    assert "future_price_leakage" in names or "target_contamination" in names


def test_audit_passes_on_clean_features(bundle, features):
    lab = build_labels(bundle.target, [20])[20]
    X = features.frame[features.model_columns()]
    audit = run_audit(X.iloc[:-25], lab["fwd_ret"].iloc[:-25], bundle.target["close"], X.index[-26],
                      pit=bundle.pit, decision_times=features.decision_times)
    assert audit.passed, audit.summary()
    # revision risk is reported (not hidden) for latest-vintage macro data
    assert any(c.name == "revision_risk" and c.status == "WARN" for c in audit.checks)


def test_backtest_stops_on_leakage(bundle, features, regimes, settings):
    from vusa_quant_ai.backtesting.walkforward import run_walkforward_backtest
    from vusa_quant_ai.features.builder import FeatureSet

    leaky = features.frame.copy()
    leaky["leak"] = bundle.target["close"].shift(-20) / bundle.target["close"] - 1
    fs2 = FeatureSet(leaky, {**features.groups, "technical": features.groups["technical"] + ["leak"]}, features.lineage,
                     features.decision_times, features.version, features.dataset_version)
    with pytest.raises(LeakageError) as e:
        run_walkforward_backtest(fs2, bundle.target, regimes, settings, None, 20, robustness=False)
    assert "BACKTEST STOPPED" in str(e.value)


def test_preprocessor_fitted_on_train_only():
    from vusa_quant_ai.models.base import Preprocessor

    X = pd.DataFrame({"a": np.r_[np.zeros(100), np.full(100, 1000.0)]})
    p = Preprocessor().fit(X.iloc[:100])
    assert p.fit_last_index_ == 99
    assert p.med_["a"] == 0.0


def test_news_leakage_check():
    from vusa_quant_ai.validation.leakage import check_news_leakage

    arts = pd.DataFrame({"published_at": ["2026-10-01T10:00:00+00:00", "2026-10-02T10:00:00+00:00"]})
    assert check_news_leakage(arts, pd.Timestamp("2026-10-03", tz="UTC")).status == "PASS"
    assert check_news_leakage(arts, pd.Timestamp("2026-10-01T12:00", tz="UTC")).status == "FAIL"
