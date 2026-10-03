"""Provider parsing tests with canned responses in each provider's real format (no network)."""

import pandas as pd
import pytest

import vusa_quant_ai.data.macro.fred as fred
import vusa_quant_ai.data.market.providers as prov
import vusa_quant_ai.data.news.ingest as ingest
from vusa_quant_ai.data.news.intelligence import (aggregate_sentiment, analyze_articles, detect_contradictions,
                                                  extract_events, lexicon_sentiment, relevance_score, source_tier)
from vusa_quant_ai.data.point_in_time.engine import PointInTimeStore
from vusa_quant_ai.data.sources.base import ProviderUnavailable


class _Resp:
    def __init__(self, text="", js=None, content=None):
        self.text, self._js = text, js
        self.content = content if content is not None else text.encode()

    def json(self):
        return self._js


def test_stooq_parsing(monkeypatch):
    csv = "Date,Open,High,Low,Close,Volume\n2024-01-02,100,101,99,100.5,1000\n2024-01-03,100.5,102,100,101.5,1200\n"
    monkeypatch.setattr(prov, "http_get", lambda *a, **k: _Resp(csv))
    df = prov.StooqProvider().history("VUSA.DE", "2024-01-01", exchange="XETRA")
    assert list(df.columns[:6]) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert df["close"].iloc[-1] == 101.5
    assert pd.Timestamp(df["available_at"].iloc[0]).tz is not None
    assert prov.StooqProvider().to_stooq("VUSA.L") == "vusa.uk"
    assert prov.StooqProvider().to_stooq("^GSPC") == "^spx"
    monkeypatch.setattr(prov, "http_get", lambda *a, **k: _Resp("No data"))
    with pytest.raises(ProviderUnavailable):
        prov.StooqProvider().history("XXX", "2024-01-01")


def test_twelvedata_and_alphavantage_errors_reported(monkeypatch):
    with pytest.raises(ProviderUnavailable, match="key not configured"):
        prov.TwelveDataProvider("").history("VUSA", "2024-01-01")
    monkeypatch.setattr(prov, "http_get", lambda *a, **k: _Resp(js={"Note": "rate limit"}))
    with pytest.raises(ProviderUnavailable, match="rate limit"):
        prov.AlphaVantageProvider("k").history("VUSA.DE", "2024-01-01")


def test_alfred_vintages_parsing(monkeypatch):
    js = {"observations": [
        {"realtime_start": "2024-02-13", "realtime_end": "2024-03-11", "date": "2024-01-01", "value": "308.4"},
        {"realtime_start": "2024-03-12", "realtime_end": "9999-12-31", "date": "2024-01-01", "value": "308.7"},
        {"realtime_start": "2024-03-12", "realtime_end": "9999-12-31", "date": "2024-02-01", "value": "310.3"},
        {"realtime_start": "2024-03-12", "realtime_end": "9999-12-31", "date": "2024-03-01", "value": "."}]}
    monkeypatch.setattr(fred, "http_get", lambda *a, **k: _Resp(js=js))
    df = fred.FredApiProvider("key").vintages("CPIAUCSL")
    assert len(df) == 3
    st = PointInTimeStore()
    st.add_records("CPIAUCSL", df, "vintage", "FRED/ALFRED")
    assert st.value_as_of("CPIAUCSL", "2024-02-20")[0] == 308.4  # first vintage
    assert st.history_as_of("CPIAUCSL", "2024-03-13").loc["2024-01-01"] == 308.7  # revised


def test_fred_csv_and_loader_fallback(monkeypatch):
    def fake_get(url, **k):
        if "api.stlouisfed" in url:
            raise ConnectionError("blocked")
        return _Resp("DATE,UNRATE\n2024-01-01,3.7\n2024-02-01,3.9\n2024-03-01,.\n")

    monkeypatch.setattr(fred, "http_get", fake_get)
    st = PointInTimeStore()
    status = fred.load_macro_into_store(st, ["UNRATE"], "key", ["fred_api", "fred_csv"], 5, "2020-01-01")
    assert status["UNRATE"].startswith("ok (latest vintage")
    assert st.method("UNRATE") == "release_lag"
    status = fred.load_macro_into_store(PointInTimeStore(), ["UNRATE"], "", ["fred_api"], 5, "2020-01-01")
    assert status["UNRATE"].startswith("unavailable")


RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Fed holds rates steady, signals cuts later this year</title><link>https://www.federalreserve.gov/a</link>
<pubDate>Wed, 30 Sep 2026 18:00:00 GMT</pubDate><description>The FOMC decision kept the policy rate unchanged.</description></item>
<item><title>Stocks plunge as inflation rises more than expected</title><link>https://www.cnbc.com/b</link>
<pubDate>Thu, 01 Oct 2026 13:00:00 GMT</pubDate><description>CPI above expectations; S&amp;P 500 sell-off deepens.</description></item>
<item><title>Stocks plunge as inflation rises more than expected!</title><link>https://www.investing.com/c</link>
<pubDate>Thu, 01 Oct 2026 13:05:00 GMT</pubDate><description>dup</description></item>
<item><title>Local bakery opens new store</title><link>https://example.com/d</link>
<pubDate>Thu, 01 Oct 2026 10:00:00 GMT</pubDate><description>Bread.</description></item>
</channel></rss>"""


def test_rss_pipeline(monkeypatch):
    monkeypatch.setattr(ingest, "http_get", lambda *a, **k: _Resp(content=RSS.encode()))
    arts = ingest.fetch_rss("https://www.federalreserve.gov/feeds/press_all.xml")
    assert len(arts) == 4 and all(a.timestamp_reliable for a in arts)
    analyze_articles(arts)
    dups = [a for a in arts if a.duplicate_of]
    assert len(dups) == 1 and "investing" in dups[0].source  # lower-tier copy is the duplicate
    bakery = [a for a in arts if "bakery" in a.title][0]
    assert bakery.relevance == 0
    infl = [a for a in arts if a.title.startswith("Stocks plunge") and not a.duplicate_of][0]
    assert infl.label == "NEGATIVE"
    ev = extract_events(arts)
    cats = {e.category for e in ev}
    assert "Monetary Policy" in cats and "Inflation" in cats
    agg = aggregate_sentiment(arts)
    assert agg["n"] >= 2 and -1 <= agg["score"] <= 1
    assert isinstance(detect_contradictions(arts), list)


def test_finance_lexicon_direction_rules():
    assert lexicon_sentiment("Inflation rises sharply, yields jump")[0] < 0
    assert lexicon_sentiment("Inflation cools as CPI comes in below expectations")[0] > 0
    assert lexicon_sentiment("Earnings beat estimates; stocks rally to record high")[0] > 0
    assert lexicon_sentiment("Company did not miss estimates")[0] >= 0
    assert relevance_score("Federal Reserve FOMC decision on interest rates")[0] > 0.5
    assert source_tier("federalreserve.gov")[0] == 1 and source_tier("random-blog.net")[0] == 5


def test_newsapi_requires_key():
    with pytest.raises(ProviderUnavailable):
        ingest.fetch_newsapi("", "q", 3)
