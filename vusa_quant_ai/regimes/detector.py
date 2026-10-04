"""Market-regime detection.

Three complementary, *causal* detectors:

1. Rule-based classifier (trend x volatility x drawdown) -> 8 interpretable regimes
2. Gaussian Hidden Markov Model on (return, volatility) - fitted on an expanding window
   and evaluated with the **forward (filtering) algorithm only**. hmmlearn's
   ``predict``/``predict_proba`` use forward-backward smoothing, which looks into the
   future, so they are deliberately not used for historical states.
3. Gaussian mixture clustering on a state vector (trend, vol, breadth, credit)

The consensus regime and the probability of each HMM state are exported as features
(regime Option B) and used to train regime-specific models (Option A).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

REGIMES = ["Strong bull", "Bull", "Neutral", "Bear", "Strong bear", "High volatility", "Crisis", "Recovery"]
REGIME_SCORE = {"Strong bull": 90, "Bull": 72, "Recovery": 62, "Neutral": 50, "High volatility": 38, "Bear": 28,
                "Strong bear": 15, "Crisis": 8}


def rule_based_regime(close: pd.Series, vix: pd.Series | None = None) -> pd.DataFrame:
    r = np.log(close).diff()
    rv20 = r.rolling(20).std() * np.sqrt(252)
    rv_pct = rv20.rolling(1260, min_periods=250).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    s50, s200 = close.rolling(50).mean(), close.rolling(200).mean()
    dd = close / close.cummax() - 1
    ret60, ret120 = close.pct_change(60), close.pct_change(120)
    lo120 = close.rolling(120).min()
    out = pd.Series("Neutral", index=close.index, dtype=object)
    up = (close > s200) & (s50 > s200)
    down = (close < s200) & (s50 < s200)
    out[up] = "Bull"
    out[up & (ret120 > 0.08) & (rv_pct < 0.6) & (dd > -0.05)] = "Strong bull"
    out[down] = "Bear"
    out[down & (dd < -0.20)] = "Strong bear"
    recov = (close / lo120 - 1 > 0.10) & (dd < -0.07) & (ret60 > 0.06) & (close > s50)
    out[recov & ~up] = "Recovery"
    hv = rv_pct > 0.85
    out[hv & ~down] = "High volatility"
    crisis = ((dd < -0.20) & (rv20 > 0.35))
    if vix is not None:
        crisis |= (vix.reindex(close.index) > 40)
    out[crisis] = "Crisis"
    out[s200.isna()] = None
    return pd.DataFrame({"regime_rule": out, "regime_score": out.map(REGIME_SCORE), "rv20": rv20, "rv_pct": rv_pct,
                         "drawdown": dd})


class CausalGaussianHMM:
    """Gaussian HMM fitted with hmmlearn (EM) but evaluated with forward filtering."""

    def __init__(self, n_states: int = 3, seed: int = 0):
        self.n_states, self.seed = n_states, seed

    def fit(self, Z: np.ndarray) -> "CausalGaussianHMM":
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:  # e.g. no hmmlearn wheel for this Python version
            return self._fit_gmm_fallback(Z)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GaussianHMM(self.n_states, covariance_type="full", n_iter=200, random_state=self.seed, tol=1e-3)
            m.fit(Z)
        self.method_ = "hmmlearn EM"
        return self._store(m.startprob_, m.transmat_, m.means_, m.covars_)

    def _store(self, startprob, transmat, means, covars) -> "CausalGaussianHMM":
        order = np.argsort(means[:, 1])  # sort states by volatility feature: 0 calm .. n-1 stressed
        self.startprob_ = startprob[order]
        self.transmat_ = transmat[np.ix_(order, order)]
        self.means_ = means[order]
        self.covars_ = covars[order]
        return self

    def _fit_gmm_fallback(self, Z: np.ndarray) -> "CausalGaussianHMM":
        """Fallback without hmmlearn: Gaussian-mixture emissions (scikit-learn) and a transition
        matrix estimated from consecutive state assignments (Laplace-smoothed). Still evaluated
        with the causal forward filter."""
        from sklearn.mixture import GaussianMixture

        g = GaussianMixture(self.n_states, covariance_type="full", random_state=self.seed, n_init=3).fit(Z)
        lab = g.predict(Z)
        K = self.n_states
        counts = np.ones((K, K))
        for a, b in zip(lab[:-1], lab[1:]):
            counts[a, b] += 1
        transmat = counts / counts.sum(axis=1, keepdims=True)
        self.method_ = "GMM + empirical transitions (hmmlearn unavailable)"
        return self._store(g.weights_, transmat, g.means_, g.covariances_)

    def filter(self, Z: np.ndarray) -> np.ndarray:
        """P(state_t | z_1..z_t) for every t (no look-ahead)."""
        K = self.n_states
        ll = np.column_stack([multivariate_normal(self.means_[k], self.covars_[k], allow_singular=True).logpdf(Z)
                              for k in range(K)])
        if ll.ndim == 1:
            ll = ll.reshape(1, -1)
        logA = np.log(self.transmat_ + 1e-12)
        alpha = np.zeros_like(ll)
        alpha[0] = np.log(self.startprob_ + 1e-12) + ll[0]
        alpha[0] -= logsumexp(alpha[0])
        for t in range(1, len(Z)):
            alpha[t] = logsumexp(alpha[t - 1][:, None] + logA, axis=0) + ll[t]
            alpha[t] -= logsumexp(alpha[t])
        return np.exp(alpha)


def hmm_state_features(close: pd.Series, n_states: int = 3, refit_every: int = 252, min_train: int = 750,
                       seed: int = 0) -> pd.DataFrame:
    r = np.log(close).diff()
    vol = r.rolling(10).std() * np.sqrt(252)
    Z = pd.DataFrame({"r": r * 100, "v": np.log(vol + 1e-4)}).dropna()
    out = pd.DataFrame(np.nan, index=close.index, columns=[f"hmm_p{k}" for k in range(n_states)])
    if len(Z) < min_train + 20:
        return out
    for start in range(min_train, len(Z), refit_every):
        model = CausalGaussianHMM(n_states, seed).fit(Z.values[:start])
        end = min(len(Z), start + refit_every)
        # filter from the beginning so the state at t uses all data up to t with params fitted up to `start`
        probs = model.filter(Z.values[:end])
        out.loc[Z.index[start:end]] = probs[start:end]
    out["hmm_stress_prob"] = out[f"hmm_p{n_states - 1}"]
    return out


def gmm_clusters(state: pd.DataFrame, n: int = 4, seed: int = 0, train_end=None) -> pd.Series:
    from sklearn.mixture import GaussianMixture

    S = state.dropna()
    if len(S) < 300:
        return pd.Series(np.nan, index=state.index)
    train = S.loc[:train_end] if train_end is not None else S
    mu, sd = train.mean(), train.std().replace(0, 1)
    g = GaussianMixture(n, covariance_type="full", random_state=seed).fit(((train - mu) / sd).values)
    lab = pd.Series(g.predict(((S - mu) / sd).values), index=S.index)
    return lab.reindex(state.index)


def detect_regimes(close: pd.Series, features: pd.DataFrame | None = None, vix: pd.Series | None = None,
                   seed: int = 0) -> pd.DataFrame:
    rb = rule_based_regime(close, vix)
    hmm = hmm_state_features(close, seed=seed)
    df = rb.join(hmm)
    if features is not None:
        cols = [c for c in ("dist_sma200", "rv_20", "sector_pct_above_sma200_proxy", "credit_risk_appetite", "vix_level")
                if c in features.columns]
        if len(cols) >= 2:
            df["gmm_cluster"] = gmm_clusters(features[cols], seed=seed)
    return df


def current_regime_summary(reg: pd.DataFrame) -> dict:
    last = reg.dropna(subset=["regime_rule"]).iloc[-1]
    rr = reg["regime_rule"].dropna()
    run = 1
    for v in rr.iloc[::-1].iloc[1:]:
        if v != rr.iloc[-1]:
            break
        run += 1
    stress = float(last.get("hmm_stress_prob", np.nan))
    vol_label = "LOW" if last["rv_pct"] < 0.33 else ("MODERATE" if last["rv_pct"] < 0.75 else "HIGH")
    prev = rr.iloc[-run - 1] if len(rr) > run else None
    return {"regime": last["regime_rule"], "score": float(last["regime_score"]), "volatility_label": vol_label,
            "realized_vol_20d": float(last["rv20"]), "vol_percentile": float(last["rv_pct"]),
            "drawdown": float(last["drawdown"]), "hmm_stress_probability": stress, "days_in_regime": run,
            "previous_regime": prev, "date": str(reg.dropna(subset=["regime_rule"]).index[-1].date()),
            "distribution_1y": rr.tail(252).value_counts(normalize=True).round(3).to_dict()}
