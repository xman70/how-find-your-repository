"""Sentiment features from the application's own news memory.

Historical sentiment only exists from the day the application started collecting news
(free feeds do not provide a historical archive). Sentiment features are therefore
included in model training only once coverage is sufficient; until then they are used
only in the current-day decision engine, and the ablation study reports that the value
of news "cannot yet be tested" rather than assuming it helps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_COVERAGE_DAYS = 250


def daily_sentiment_from_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stored articles to one row per *availability* date (publication date, UTC)."""
    if articles is None or articles.empty:
        return pd.DataFrame(columns=["score", "dispersion", "n"])
    a = articles.copy()
    a = a[a["duplicate_of"].isna() & (a["relevance"].astype(float) >= 0.2)]
    if a.empty:
        return pd.DataFrame(columns=["score", "dispersion", "n"])
    a["published_at"] = pd.to_datetime(a["published_at"], utc=True)
    a["w"] = a["relevance"].astype(float) * a["source_quality"].astype(float)
    a["day"] = a["published_at"].dt.tz_convert(None).dt.normalize()
    g = a.groupby("day")
    out = pd.DataFrame({
        "score": g.apply(lambda x: np.average(x["sentiment"].astype(float), weights=x["w"] + 1e-9), include_groups=False),
        "n": g.size(),
    })
    out["dispersion"] = g.apply(lambda x: float(np.std(x["sentiment"].astype(float))), include_groups=False)
    out["available_at"] = (out.index + pd.Timedelta(days=1)).tz_localize("UTC")  # conservative: next day
    return out


def sentiment_features(daily: pd.DataFrame, decision_times: pd.DatetimeIndex, index: pd.DatetimeIndex
                       ) -> tuple[pd.DataFrame, dict[str, str], int]:
    if daily is None or daily.empty:
        return pd.DataFrame(index=index), {}, 0
    from ...data.point_in_time.engine import align_market_to_decision

    d = daily.rename(columns={"score": "close"})
    m = align_market_to_decision(d, decision_times)
    s = pd.Series(m["close"].values, index=index)
    disp = pd.Series(m["dispersion"].values, index=index)
    f = pd.DataFrame({
        "news_sentiment": s,
        "news_sentiment_ma5": s.rolling(5, min_periods=1).mean(),
        "news_sentiment_mom": s.rolling(5, min_periods=2).mean() - s.rolling(20, min_periods=5).mean(),
        "news_sentiment_dispersion": disp,
    }, index=index)
    coverage = int(s.notna().sum())
    return f, {c: "news memory (lexicon/FinBERT)" for c in f.columns}, coverage
