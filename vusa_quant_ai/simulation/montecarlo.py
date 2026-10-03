"""MONTE CARLO 2.0 - six path generators compared side by side.

1. Geometric Brownian motion (Gaussian benchmark - known to understate tails)
2. Historical i.i.d. bootstrap
3. Stationary block bootstrap (preserves short-range dependence / vol clustering)
4. Regime-switching simulation (2-state Markov chain with state-specific return pools)
5. Volatility-cluster simulation (GJR-GARCH(1,1) with Student-t innovations, MLE)
6. Model-driven simulation (centre = ensemble forecast drift, dispersion = conformal residuals)

All simulations are clearly *scenarios of plausible paths*, not predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

PCTS = (10, 25, 50, 75, 90)


@dataclass
class MCResult:
    method: str
    horizon: int
    n_paths: int
    terminal_returns: np.ndarray
    fan: pd.DataFrame  # percentiles of price path per step
    stats: dict


def _summarise(method: str, paths: np.ndarray, s0: float, sample_paths: int = 0) -> MCResult:
    # paths: (n, h) cumulative log returns
    prices = s0 * np.exp(paths)
    fan = pd.DataFrame({f"P{p}": np.percentile(prices, p, axis=0) for p in PCTS})
    fan.index = np.arange(1, paths.shape[1] + 1)
    term = np.expm1(paths[:, -1])
    running_max = np.maximum.accumulate(np.concatenate([np.zeros((paths.shape[0], 1)), paths], axis=1), axis=1)
    mdd = (np.exp(np.concatenate([np.zeros((paths.shape[0], 1)), paths], axis=1) - running_max) - 1).min(axis=1)
    st = {f"P{p}": float(np.percentile(term, p)) for p in PCTS}
    st.update({"mean": float(term.mean()), "p_up": float((term > 0).mean()), "p_gt5": float((term > 0.05).mean()),
               "p_lt5": float((term < -0.05).mean()), "p_lt10": float((term < -0.10).mean()),
               "VaR95": float(-np.percentile(term, 5)), "CVaR95": float(-term[term <= np.percentile(term, 5)].mean()),
               "p_maxdd_gt10": float((mdd < -0.10).mean()), "median_maxdd": float(np.median(mdd)),
               "price_P10": float(s0 * (1 + st["P10"])), "price_P50": float(s0 * (1 + st["P50"])),
               "price_P90": float(s0 * (1 + st["P90"]))})
    return MCResult(method, paths.shape[1], paths.shape[0], term, fan, st)


def gbm(r: np.ndarray, h: int, n: int, rng) -> np.ndarray:
    mu, sd = r.mean(), r.std()
    return np.cumsum(rng.normal(mu, sd, (n, h)), axis=1)


def iid_bootstrap(r: np.ndarray, h: int, n: int, rng) -> np.ndarray:
    return np.cumsum(r[rng.integers(0, len(r), (n, h))], axis=1)


def block_bootstrap(r: np.ndarray, h: int, n: int, rng, mean_block: int = 20) -> np.ndarray:
    """Politis-Romano stationary bootstrap."""
    L = len(r)
    out = np.empty((n, h))
    pos = rng.integers(0, L, n)
    p = 1 / mean_block
    for t in range(h):
        out[:, t] = r[pos]
        jump = rng.random(n) < p
        pos = np.where(jump, rng.integers(0, L, n), (pos + 1) % L)
    return np.cumsum(out, axis=1)


def regime_switching(r: np.ndarray, h: int, n: int, rng, current_stress_prob: float | None = None) -> np.ndarray:
    rv = pd.Series(r).rolling(20, min_periods=10).std().bfill().values
    thr = np.quantile(rv, 0.75)
    state = (rv > thr).astype(int)
    P = np.zeros((2, 2))
    for a, b in zip(state[:-1], state[1:]):
        P[a, b] += 1
    P = (P + 1) / (P + 1).sum(axis=1, keepdims=True)
    pools = [r[state == 0], r[state == 1]]
    s = np.full(n, state[-1]) if current_stress_prob is None else (rng.random(n) < current_stress_prob).astype(int)
    out = np.empty((n, h))
    for t in range(h):
        for k in (0, 1):
            m = s == k
            out[m, t] = pools[k][rng.integers(0, len(pools[k]), m.sum())]
        s = np.where(rng.random(n) < P[s, 1], 1, 0)
    return np.cumsum(out, axis=1)


def _garch_path(x, omega, a, g, b):
    """Conditional variances h_t = omega + (a + g*1[x_{t-1}<0]) x_{t-1}^2 + b h_{t-1} (vectorised)."""
    from scipy.signal import lfilter

    u = np.empty(len(x))
    u[0] = x.var() * (1 - b)
    u[1:] = omega + (a + g * (x[:-1] < 0)) * x[:-1] ** 2
    return lfilter([1.0], [1.0, -b], u, zi=[b * x.var()])[0]


def fit_gjr_garch(r: np.ndarray) -> dict:
    """MLE of GJR-GARCH(1,1) with Student-t innovations (pure numpy/scipy)."""
    x = r - r.mean()

    def nll(p):
        omega, a, g, b, nu = p
        if omega <= 0 or a < 0 or b < 0 or a + b + g / 2 >= 0.999 or nu <= 2.1 or a + g < 0:
            return 1e10
        h = _garch_path(x, omega, a, g, b)
        sc = np.sqrt(h * (nu - 2) / nu)
        return -np.sum(stats.t.logpdf(x / sc, nu) - np.log(sc))

    v = x.var()
    res = optimize.minimize(nll, [v * 0.05, 0.05, 0.08, 0.85, 6.0], method="Nelder-Mead",
                            options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-6})
    omega, a, g, b, nu = res.x
    h = float(_garch_path(x, omega, a, g, b)[-1])
    return {"omega": omega, "alpha": a, "gamma": g, "beta": b, "nu": nu, "mu": r.mean(), "h_last": h,
            "x_last": x[-1], "converged": bool(res.success)}


def garch_sim(r: np.ndarray, h: int, n: int, rng, params: dict | None = None) -> np.ndarray:
    p = params or fit_gjr_garch(r[-2000:])
    nu = p["nu"]
    hv = np.full(n, p["omega"] + (p["alpha"] + p["gamma"] * (p["x_last"] < 0)) * p["x_last"] ** 2 + p["beta"] * p["h_last"])
    out = np.empty((n, h))
    for t in range(h):
        z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
        e = np.sqrt(hv) * z
        out[:, t] = p["mu"] + e
        hv = p["omega"] + (p["alpha"] + p["gamma"] * (e < 0)) * e ** 2 + p["beta"] * hv
    return np.cumsum(out, axis=1)


def model_driven(r: np.ndarray, h: int, n: int, rng, forecast_ret: float, resid: np.ndarray | None) -> np.ndarray:
    """Block-bootstrap path shapes re-centred so the terminal distribution matches the model
    forecast (drift) and, if available, the empirical conformal residual dispersion."""
    paths = block_bootstrap(r, h, n, rng)
    term = paths[:, -1]
    target_mu = np.log1p(forecast_ret)
    if resid is not None and len(resid) > 30:
        target_sd = np.std(np.log1p(np.clip(forecast_ret + resid, -0.95, None)))
        scale = target_sd / (term.std() or 1)
    else:
        scale = 1.0
    steps = np.arange(1, h + 1) / h
    centred = (paths - term.mean() * steps) * scale
    return centred + target_mu * steps


def run_monte_carlo(close: pd.Series, horizon: int = 20, n_paths: int = 10000, seed: int = 42,
                    forecast_ret: float | None = None, conformal_resid: np.ndarray | None = None,
                    stress_prob: float | None = None, lookback: int = 2520) -> dict[str, MCResult]:
    rng = np.random.default_rng(seed)
    r = np.log(close).diff().dropna().values[-lookback:]
    s0 = float(close.iloc[-1])
    out = {
        "Geometric Brownian Motion": _summarise("gbm", gbm(r, horizon, n_paths, rng), s0),
        "Historical bootstrap": _summarise("bootstrap", iid_bootstrap(r, horizon, n_paths, rng), s0),
        "Block bootstrap": _summarise("block", block_bootstrap(r, horizon, n_paths, rng), s0),
        "Regime switching": _summarise("regime", regime_switching(r, horizon, n_paths, rng, stress_prob), s0),
        "GJR-GARCH vol clustering": _summarise("garch", garch_sim(r, horizon, n_paths, rng), s0),
    }
    if forecast_ret is not None and np.isfinite(forecast_ret):
        out["Model-driven"] = _summarise("model", model_driven(r, horizon, n_paths, rng, forecast_ret, conformal_resid), s0)
    return out


def mc_comparison_table(res: dict[str, MCResult]) -> pd.DataFrame:
    return pd.DataFrame({k: v.stats for k, v in res.items()}).T
