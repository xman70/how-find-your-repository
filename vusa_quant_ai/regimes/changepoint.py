"""Change-point and structural-break detection (all causal / online).

* CUSUM on standardised returns (mean shift)
* Page-Hinkley on squared returns (volatility shift)
* Bayesian online change-point detection (Adams & MacKay 2007) with a Normal-Gamma prior
* Rolling two-sample Kolmogorov-Smirnov test: recent window vs long history
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln


def cusum(x: pd.Series, k: float = 0.5, h: float = 5.0, warmup: int = 250) -> pd.DataFrame:
    x = x.dropna()
    mu = x.expanding(warmup).mean().shift(1)
    sd = x.expanding(warmup).std().shift(1)
    z = ((x - mu) / sd).fillna(0).values
    sp = sn = 0.0
    up, dn, alarm = np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z), dtype=bool)
    for t, zt in enumerate(z):
        sp = max(0.0, sp + zt - k)
        sn = max(0.0, sn - zt - k)
        up[t], dn[t] = sp, sn
        if sp > h or sn > h:
            alarm[t] = True
            sp = sn = 0.0
    return pd.DataFrame({"cusum_pos": up, "cusum_neg": dn, "cusum_alarm": alarm}, index=x.index)


def page_hinkley(x: pd.Series, delta: float = 0.005, lam: float = 50.0, alpha: float = 0.999) -> pd.DataFrame:
    x = x.dropna()
    mean, m_t, M_t = 0.0, 0.0, 0.0
    ph, alarm = np.zeros(len(x)), np.zeros(len(x), dtype=bool)
    v = x.values
    scale = np.nanstd(v[:250]) if len(v) > 250 else np.nanstd(v) or 1.0
    n = 0
    for t, xt in enumerate(v / (scale or 1.0)):
        n += 1
        mean = mean + (xt - mean) / n
        m_t = alpha * m_t + (xt - mean - delta)
        M_t = min(M_t, m_t)
        ph[t] = m_t - M_t
        if ph[t] > lam:
            alarm[t] = True
            m_t, M_t, mean, n = 0.0, 0.0, 0.0, 0
    return pd.DataFrame({"ph_stat": ph, "ph_alarm": alarm}, index=x.index)


def bocpd(x: pd.Series, hazard: float = 1 / 250, max_run: int = 500, mu0: float = 0.0, kappa0: float = 1.0,
          alpha0: float = 1.0, beta0: float = 1.0) -> pd.DataFrame:
    """Bayesian online change-point detection; returns P(run length < 5) and the MAP run length."""
    v = x.dropna().values
    v = (v - np.nanmean(v[:250])) / (np.nanstd(v[:250]) or 1.0) if len(v) > 250 else v
    R = np.array([1.0])
    mu, kappa, alpha, beta = np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0])
    cp_prob, map_rl = np.zeros(len(v)), np.zeros(len(v))
    for t, xt in enumerate(v):
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        logpred = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df) - np.log(scale)
                   - (df + 1) / 2 * np.log1p(((xt - mu) / scale) ** 2 / df))
        pred = np.exp(logpred)
        growth = R * pred * (1 - hazard)
        cp = (R * pred * hazard).sum()
        R = np.concatenate([[cp], growth])
        R /= R.sum()
        mu_n = (kappa * mu + xt) / (kappa + 1)
        kappa_n = kappa + 1
        alpha_n = alpha + 0.5
        beta_n = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_n])
        kappa = np.concatenate([[kappa0], kappa_n])
        alpha = np.concatenate([[alpha0], alpha_n])
        beta = np.concatenate([[beta0], beta_n])
        if len(R) > max_run:
            R, mu, kappa, alpha, beta = R[:max_run], mu[:max_run], kappa[:max_run], alpha[:max_run], beta[:max_run]
            R /= R.sum()
        cp_prob[t] = R[:5].sum()
        map_rl[t] = np.argmax(R)
    idx = x.dropna().index
    return pd.DataFrame({"bocpd_recent_cp_prob": cp_prob, "bocpd_map_run_length": map_rl}, index=idx)


def rolling_ks(x: pd.Series, recent: int = 126, history: int = 1260, step: int = 5) -> pd.DataFrame:
    v = x.dropna()
    stat, p = pd.Series(np.nan, index=v.index), pd.Series(np.nan, index=v.index)
    for t in range(recent + 250, len(v), step):
        a = v.values[t - recent + 1: t + 1]
        b = v.values[max(0, t - recent - history + 1): t - recent + 1]
        if len(b) > 100:
            s, pv = stats.ks_2samp(a, b)
            stat.iloc[t], p.iloc[t] = s, pv
    return pd.DataFrame({"ks_stat": stat.ffill(limit=step), "ks_p": p.ffill(limit=step)})


def structural_change_report(close: pd.Series) -> dict:
    r = np.log(close).diff().dropna()
    cs = cusum(r)
    ph = page_hinkley(r ** 2)
    bo = bocpd(r.tail(1500))
    ks = rolling_ks(r)
    ks_vol = rolling_ks(r.abs())
    last_alarm = lambda s: str(s[s].index[-1].date()) if s.any() else None  # noqa: E731
    recent = lambda s, n=21: bool(s.tail(n).any())  # noqa: E731
    rep = {
        "cusum_last_alarm": last_alarm(cs["cusum_alarm"]), "cusum_recent": recent(cs["cusum_alarm"]),
        "page_hinkley_last_alarm": last_alarm(ph["ph_alarm"]), "page_hinkley_recent": recent(ph["ph_alarm"]),
        "bocpd_cp_prob_now": float(bo["bocpd_recent_cp_prob"].iloc[-1]),
        "bocpd_max_cp_prob_21d": float(bo["bocpd_recent_cp_prob"].tail(21).max()),
        "bocpd_run_length": int(bo["bocpd_map_run_length"].iloc[-1]),
        "ks_returns_p": float(ks["ks_p"].dropna().iloc[-1]) if ks["ks_p"].notna().any() else np.nan,
        "ks_abs_returns_p": float(ks_vol["ks_p"].dropna().iloc[-1]) if ks_vol["ks_p"].notna().any() else np.nan,
    }
    rep["recent_6m_differs"] = bool((rep["ks_abs_returns_p"] < 0.01) or (rep["ks_returns_p"] < 0.01))
    rep["structural_change_detected"] = bool(rep["cusum_recent"] or rep["page_hinkley_recent"] or
                                             rep["bocpd_max_cp_prob_21d"] > 0.5 or rep["recent_6m_differs"])
    rep["series"] = {"cusum": cs, "page_hinkley": ph, "bocpd": bo, "ks": ks}
    return rep
