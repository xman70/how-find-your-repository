"""Learned traffic-light model: adopted only when it genuinely helps out of sample."""
import numpy as np
import pandas as pd

from vusa_quant_ai.signals import light_model as LM
from vusa_quant_ai.signals.terms import RECON


def _data(signal_strength: float, n: int = 2000, seed: int = 0):
    """Components on sampled rows (every 5 sessions); 'volatility' predicts the outcome, but the
    hand-set score mostly ignores it."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n * 5)[::5]
    C = pd.DataFrame(rng.normal(50, 15, (n, len(RECON))), index=idx, columns=RECON)
    vol = C["volatility"].to_numpy()
    y = pd.Series(signal_strength * (vol - 50) / 15 * 0.02 + rng.normal(0, 0.02, n), index=idx)
    heur = 0.95 * C["trend"] + 0.05 * C["volatility"]  # hand-set light dominated by an useless input
    return C, y, heur


def test_learner_adopted_when_signal_is_real_and_holds_out_of_sample():
    C, y, heur = _data(signal_strength=0.6)
    p, final, coefs = LM.walk_forward(C, y, h=20, step=5)
    assert p.notna().sum() > 1000 and coefs["volatility"] > 0.2  # found the right driver
    learned = pd.Series(LM.learned_score(p), index=C.index)
    sel = LM.select_blend(heur, learned, y, h=20, step=5)
    assert sel["alpha"] > 0, sel["reason"]
    assert sel["holdout_auc_chosen"] > sel["holdout_auc_handset"] + 0.05  # untouched hold-out improves


def test_learner_rejected_on_pure_noise():
    C, y, heur = _data(signal_strength=0.0, seed=3)
    p, _, _ = LM.walk_forward(C, y, h=20, step=5)
    sel = LM.select_blend(heur, pd.Series(LM.learned_score(p), index=C.index), y, h=20, step=5)
    assert sel["alpha"] == 0.0


def test_learner_rejected_with_too_few_independent_periods():
    C, y, heur = _data(signal_strength=0.6)
    p, _, _ = LM.walk_forward(C, y, h=252, step=5)
    sel = LM.select_blend(heur, pd.Series(LM.learned_score(p), index=C.index), y, h=252, step=5)
    assert sel["alpha"] == 0.0 and "too few" in sel["reason"]


def test_walk_forward_never_uses_unfinished_outcomes():
    """Predictions for a block must not change when outcomes inside the purge gap are altered."""
    C, y, _ = _data(signal_strength=0.6, n=600)
    p1, _, _ = LM.walk_forward(C, y, h=20, step=5, refit_rows=50, min_train_rows=150)
    y2 = y.copy()
    gap = int(np.ceil(20 / 5)) + 1
    start = 150 + gap  # first prediction row
    y2.iloc[start - gap:] = 1.0  # outcomes not yet known at 'start'
    p2, _, _ = LM.walk_forward(C, y2, h=20, step=5, refit_rows=50, min_train_rows=150)
    assert np.allclose(p1.iloc[start:start + 50], p2.iloc[start:start + 50])


def test_auc():
    y = pd.Series(np.arange(100.0))
    assert LM.auc(y, y) == 1.0 and LM.auc(-y, y) == 0.0
