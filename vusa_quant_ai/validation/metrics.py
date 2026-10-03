"""Forecast, probabilistic and trading metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= ~pd.isna(np.asarray(a, dtype=float))
    return [np.asarray(a, dtype=float)[m] for a in arrs]


def brier(y, p) -> float:
    y, p = _clean(y, p)
    return float(np.mean((p - y) ** 2)) if len(y) else np.nan


def log_loss(y, p, eps=1e-6) -> float:
    y, p = _clean(y, p)
    if not len(y):
        return np.nan
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = _clean(y, p)
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = ids == b
        if m.sum():
            rows.append({"bin": b, "p_mean": p[m].mean(), "freq": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def ece(y, p, bins: int = 10) -> float:
    r = reliability(y, p, bins)
    if r.empty:
        return np.nan
    return float((r["n"] * (r["p_mean"] - r["freq"]).abs()).sum() / r["n"].sum())


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    y, p = _clean(y, p)
    if not len(y):
        return {}
    pred = (p >= threshold).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec) and not np.isnan(rec) else np.nan
    auc = np.nan
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) == 2:
            auc = float(roc_auc_score(y, p))
    except Exception:
        pass
    return {"accuracy": float((pred == y).mean()), "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "brier": brier(y, p), "log_loss": log_loss(y, p), "ece": ece(y, p), "base_rate": float(y.mean()), "n": int(len(y))}


def regression_metrics(y, yhat) -> dict:
    y, yhat = _clean(y, yhat)
    if not len(y):
        return {}
    e = yhat - y
    ic = float(pd.Series(y).corr(pd.Series(yhat), method="spearman")) if len(y) > 5 else np.nan
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
            "directional_accuracy": float((np.sign(y) == np.sign(yhat)).mean()), "ic_spearman": ic, "n": int(len(y))}


def interval_metrics(y, lo, hi, nominal: float) -> dict:
    y, lo, hi = _clean(y, lo, hi)
    if not len(y):
        return {}
    cov = float(((y >= lo) & (y <= hi)).mean())
    return {"coverage": cov, "nominal": nominal, "coverage_gap": cov - nominal, "mean_width": float((hi - lo).mean())}


def trading_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    vol = r.std() * np.sqrt(periods_per_year)
    dn = r[r < 0].std() * np.sqrt(periods_per_year)
    mdd = (eq / eq.cummax() - 1).min()
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"total_return": float(eq.iloc[-1] - 1), "cagr": float(cagr), "volatility": float(vol),
            "sharpe": float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else np.nan,
            "sortino": float(r.mean() * periods_per_year / dn) if dn and dn > 0 else np.nan,
            "max_drawdown": float(mdd), "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
            "profit_factor": float(gains / losses) if losses > 0 else np.nan,
            "hit_rate": float((r > 0).mean()), "n_days": int(len(r))}


def diebold_mariano(e1, e2, h: int = 1) -> dict:
    """DM test on squared errors with Newey-West variance (HAC lag h-1). Negative stat -> model 1 better."""
    e1, e2 = _clean(e1, e2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 30:
        return {"stat": np.nan, "p_value": np.nan, "n": n}
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[: n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))) / n
    if var <= 0:
        return {"stat": np.nan, "p_value": np.nan, "n": n}
    from scipy import stats

    stat = dbar / np.sqrt(var)
    return {"stat": float(stat), "p_value": float(2 * (1 - stats.norm.cdf(abs(stat)))), "n": n}


def block_bootstrap_ci(x, stat=np.mean, n_boot: int = 1000, block: int = 20, alpha: float = 0.1, seed: int = 0):
    x = np.asarray(pd.Series(x).dropna())
    if len(x) < block * 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(x)
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, n // block + 1)
        sample = np.concatenate([x[s: s + block] for s in starts])[:n]
        out.append(stat(sample))
    return (float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2)))
