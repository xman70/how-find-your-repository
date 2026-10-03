"""News ingestion: RSS/Atom feeds and NewsAPI.

Only official feeds and APIs are used (no scraping of article pages). Each article keeps
its own publication timestamp; articles without a reliable timestamp are stamped with
the retrieval time (the earliest moment *we* could have known them), so they can never
leak into an earlier decision.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from urllib.parse import urlparse

import pandas as pd

from ...core.runtime import get_logger, utcnow
from ..sources.base import FetchResult, ProviderUnavailable, SourceHealth, http_get

log = get_logger("vusa.news")


@dataclass
class Article:
    title: str
    summary: str
    url: str
    source: str
    published_at: str
    retrieved_at: str
    timestamp_reliable: bool = True
    article_id: str = ""
    relevance: float = 0.0
    topics: list[str] = field(default_factory=list)
    source_quality: float = 0.0
    sentiment: float = 0.0
    label: str = "NEUTRAL"
    duplicate_of: str | None = None

    def __post_init__(self):
        if not self.article_id:
            self.article_id = hashlib.sha1((self.url or self.title).encode("utf-8", "ignore")).hexdigest()[:16]

    def to_row(self) -> dict:
        d = asdict(self)
        d["topics"] = ",".join(self.topics)
        d.pop("timestamp_reliable")
        return d


def _clean(html: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", txt).strip()


def fetch_rss(url: str, timeout: float = 15.0) -> list[Article]:
    try:
        import feedparser
    except ImportError as exc:
        raise ProviderUnavailable("feedparser not installed") from exc
    raw = http_get(url, timeout=timeout).content
    feed = feedparser.parse(raw)
    if feed.bozo and not feed.entries:
        raise ProviderUnavailable(f"unparseable feed: {feed.bozo_exception}")
    now = utcnow()
    host = urlparse(url).netloc.replace("www.", "")
    out = []
    for e in feed.entries:
        ts = e.get("published_parsed") or e.get("updated_parsed")
        reliable = ts is not None
        pub = pd.Timestamp(*ts[:6], tz="UTC") if reliable else pd.Timestamp(now)
        if pub > pd.Timestamp(now) + pd.Timedelta(minutes=5):  # future-dated item -> distrust
            pub, reliable = pd.Timestamp(now), False
        link = e.get("link", "")
        src = urlparse(link).netloc.replace("www.", "") or host  # attribute to the publishing domain
        out.append(Article(title=_clean(e.get("title", "")), summary=_clean(e.get("summary", ""))[:1500],
                           url=link, source=src, published_at=pub.isoformat(),
                           retrieved_at=now.isoformat(), timestamp_reliable=reliable))
    return out


def fetch_newsapi(api_key: str, query: str, lookback_days: int, timeout: float = 15.0) -> list[Article]:
    if not api_key:
        raise ProviderUnavailable("NewsAPI key not configured")
    now = utcnow()
    js = http_get("https://newsapi.org/v2/everything", timeout=timeout, params={
        "q": query, "language": "en", "sortBy": "publishedAt", "pageSize": 100,
        "from": (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d"), "apiKey": api_key}).json()
    if js.get("status") != "ok":
        raise ProviderUnavailable(f"NewsAPI: {js.get('message', 'error')}")
    out = []
    for a in js.get("articles", []):
        pub = pd.Timestamp(a.get("publishedAt") or now).tz_convert("UTC") if a.get("publishedAt") else pd.Timestamp(now)
        out.append(Article(title=a.get("title") or "", summary=(a.get("description") or "")[:1500],
                           url=a.get("url") or "", source=(a.get("source") or {}).get("name", "newsapi"),
                           published_at=pub.isoformat(), retrieved_at=now.isoformat()))
    return out


def collect_news(feeds: list[str], newsapi_key: str, priority: list[str], lookback_days: int, max_articles: int,
                 timeout: float, health: SourceHealth | None = None) -> tuple[list[Article], list[FetchResult]]:
    """Collect from every configured source. Failure of one source never blocks others."""
    import time

    articles: list[Article] = []
    attempts: list[FetchResult] = []
    cutoff = pd.Timestamp(utcnow()) - pd.Timedelta(days=lookback_days)
    if "rss" in priority:
        for url in feeds:
            t0 = time.perf_counter()
            try:
                items = fetch_rss(url, timeout)
                items = [a for a in items if pd.Timestamp(a.published_at) >= cutoff]
                articles.extend(items)
                res = FetchResult("news", urlparse(url).netloc, True, rows=len(items),
                                  latency_ms=(time.perf_counter() - t0) * 1000)
            except Exception as exc:  # noqa: BLE001
                res = FetchResult("news", urlparse(url).netloc, False, error=f"{type(exc).__name__}: {exc}"[:300],
                                  latency_ms=(time.perf_counter() - t0) * 1000)
            attempts.append(res)
            if health:
                health.record(res)
    if "newsapi" in priority:
        t0 = time.perf_counter()
        try:
            items = fetch_newsapi(newsapi_key, '"S&P 500" OR "Federal Reserve" OR inflation OR "stock market"',
                                  lookback_days, timeout)
            articles.extend(items)
            res = FetchResult("news", "newsapi", True, rows=len(items), latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            res = FetchResult("news", "newsapi", False, error=str(exc)[:300])
        attempts.append(res)
        if health:
            health.record(res)
    articles.sort(key=lambda a: a.published_at, reverse=True)
    return articles[:max_articles], attempts
