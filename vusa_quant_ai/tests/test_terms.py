"""Short-term / long-term traffic lights."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from vusa_quant_ai.config.settings import SignalSettings
from vusa_quant_ai.signals import terms as T
from vusa_quant_ai.signals.components import Component


def hr(h, p_up, base, conf=0.6, exp=0.01):
    return SimpleNamespace(horizon=h, p_up=p_up, confidence=conf, expected_return=exp, interval=(-0.05, 0.07),
                           interval_level=0.8, p_lt5=0.1, p_gt5=0.2, quantiles={}, warnings=[], oos=None,
                           model_preds={"naive_drift": {"p_up": base, "ret": 0.0}})


def comps(score):
    names = ["trend", "momentum", "valuation", "macro", "sentiment", "volatility", "breadth", "regime", "event_risk",
             "ml_forecast", "analogues", "risk_reward"]
    return {n: Component(n, score, [n]) for n in names}


def test_ml_score_is_relative_to_base_rate():
    # 75% chance of a rise over a year is ordinary when the base rate is 75%
    ordinary = T.ml_term_component([hr(252, 0.75, 0.75)]).score
    above = T.ml_term_component([hr(252, 0.90, 0.75)]).score
    below = T.ml_term_component([hr(252, 0.55, 0.75)]).score
    assert below < ordinary < above
    assert below < 50  # below-normal odds count against, even though P(up) > 50%
    assert np.isnan(T.ml_term_component([]).score)


def test_terms_use_their_own_horizons_and_weights():
    fc = SimpleNamespace(horizons={5: hr(5, 0.70, 0.55), 20: hr(20, 0.72, 0.58), 252: hr(252, 0.50, 0.76)})
    cfg = SignalSettings()
    s = T.decide_term("short", comps(60), fc, 30, cfg, None, {}, {})
    l_ = T.decide_term("long", comps(60), fc, 30, cfg, None, {}, {})
    assert s["horizons_used"] == [5, 20] and l_["horizons_used"] == [252]
    assert s["components"]["ml_forecast"]["score"] > 60 > l_["components"]["ml_forecast"]["score"]
    assert s["headline"]["horizon"] == 20 and l_["headline"]["horizon"] == 252
    assert s["color"] in ("GREEN", "YELLOW", "RED") and s["reasons_for"]
    assert T.TERMS["long"]["weights"]["valuation"] > T.TERMS["short"]["weights"]["valuation"]
    assert T.TERMS["short"]["weights"]["momentum"] > T.TERMS["long"]["weights"]["momentum"]


def test_term_without_forecast_says_so():
    d = T.decide_term("long", comps(55), SimpleNamespace(horizons={5: hr(5, 0.6, 0.55)}), 40, SignalSettings(), None, {}, {})
    assert d["headline"] is None and any("No ML forecast" in n for n in d["notes"])


def test_track_record_detects_informative_and_useless_lights(monkeypatch):
    idx = pd.bdate_range("2015-01-01", periods=1500)
    rng = np.random.default_rng(0)
    score = pd.Series(50 + 20 * np.sin(np.arange(1500) / 40), index=idx)
    for informative in (True, False):
        y = pd.Series(rng.normal(0.01, 0.03, 1500), index=idx) + (0.03 * (score - 50) / 20 if informative else 0)
        h = hr(20, 0.6, 0.58)
        h.oos = pd.DataFrame({"y_ret": y, "ens_up": 0.6}, index=idx)
        fc = SimpleNamespace(horizons={20: h})
        monkeypatch.setattr(T, "historical_term_scores", lambda *a, **k: pd.DataFrame({"score": score}))
        tr = T.light_track_record("short", fc, pd.DataFrame(index=idx), pd.DataFrame(index=idx), SignalSettings())
        assert tr["available"] and tr["horizon"] == 20
        assert (tr["informative"] is True) == informative
        assert tr["by_light"]["GREEN"]["n"] > 0 and "independent" in tr["note"]
