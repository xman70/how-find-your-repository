"""News intelligence: de-duplication, relevance, source quality, finance-domain sentiment,
structured event extraction and contradiction detection.

The sentiment model is a transparent finance-domain lexicon with negation handling and
domain-specific *direction* rules (e.g. "inflation rises" is bearish for equities even
though "rises" is a positive word). It does not merely count positive headlines: each
article contributes ``sentiment x relevance x source_quality`` and dispersion across
sources is reported. If a transformer finance model (FinBERT) is installed it can be
used instead via :func:`get_sentiment_model`.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .ingest import Article

# ---------------------------------------------------------------------- source quality
SOURCE_TIERS = {
    # tier 1: official / primary sources
    "federalreserve.gov": (1, 0.98), "ecb.europa.eu": (1, 0.97), "bls.gov": (1, 0.98), "bea.gov": (1, 0.98),
    "treasury.gov": (1, 0.97), "spglobal.com": (1, 0.95), "vanguard.com": (1, 0.95), "sec.gov": (1, 0.97),
    "imf.org": (1, 0.94), "bis.org": (1, 0.94),
    # tier 2: major financial institutions / data vendors
    "reuters.com": (2, 0.92), "bloomberg.com": (2, 0.92), "ft.com": (2, 0.9), "wsj.com": (2, 0.9),
    "a.dj.com": (2, 0.88), "dowjones.io": (2, 0.88),
    # tier 3: established financial media
    "cnbc.com": (3, 0.82), "marketwatch.com": (3, 0.8), "barrons.com": (3, 0.82), "economist.com": (3, 0.85),
    "apnews.com": (3, 0.85), "finance.yahoo.com": (3, 0.72), "yahoo.com": (3, 0.7),
    # tier 4: secondary analysis
    "investing.com": (4, 0.65), "seekingalpha.com": (4, 0.55), "fool.com": (4, 0.5), "zerohedge.com": (5, 0.3),
}
DEFAULT_TIER = (5, 0.4)


def source_tier(source: str, url: str = "") -> tuple[int, float]:
    """Tier of the article's own domain (link) first; the feed/source name only as fallback."""
    host = (urlparse(url).netloc or "").lower().replace("www.", "")
    if host:
        for dom, tier in SOURCE_TIERS.items():
            if host == dom or host.endswith("." + dom):
                return tier
    name = (source or "").lower().replace("www.", "")
    for dom, tier in SOURCE_TIERS.items():
        if name == dom or name.endswith("." + dom) or (not host and dom in name):
            return tier
    return DEFAULT_TIER


# ---------------------------------------------------------------------- topics / relevance
TOPICS = {
    "sp500": [r"s&p ?500", r"\bspx\b", r"\bspy\b", r"wall street", r"u\.?s\.? stocks?", r"equities", r"stock market",
              r"\bvusa\b", r"vanguard s&p", r"dow jones", r"nasdaq", r"blue[- ]chip"],
    "monetary_policy": [r"federal reserve", r"\bfed\b", r"fomc", r"powell", r"rate (cut|hike|decision)", r"interest rates?",
                        r"monetary policy", r"\becb\b", r"lagarde", r"basis points?", r"balance sheet", r"quantitative"],
    "inflation": [r"inflation", r"\bcpi\b", r"\bpce\b", r"consumer prices", r"price index", r"deflation", r"disinflation"],
    "employment": [r"payrolls?", r"jobless", r"unemployment", r"labor market", r"jobs report", r"hiring", r"layoffs?"],
    "growth_recession": [r"recession", r"\bgdp\b", r"economic growth", r"slowdown", r"contraction", r"soft landing",
                         r"hard landing", r"\bism\b", r"\bpmi\b"],
    "earnings": [r"earnings", r"profit", r"revenue", r"guidance", r"\beps\b", r"quarterly results", r"buyback"],
    "geopolitics": [r"\bwar\b", r"sanction", r"tariff", r"geopolit", r"conflict", r"invasion", r"election", r"trade war",
                    r"missile", r"terror"],
    "financial_conditions": [r"credit spread", r"liquidity", r"bank(ing)? (crisis|failure|stress)", r"default",
                             r"treasury yields?", r"bond yields?", r"volatility", r"\bvix\b", r"margin call", r"dollar"],
}
TOPIC_WEIGHT = {"sp500": 1.0, "monetary_policy": 0.9, "inflation": 0.8, "employment": 0.7, "growth_recession": 0.8,
                "earnings": 0.7, "geopolitics": 0.6, "financial_conditions": 0.75}


def classify_topics(text: str) -> dict[str, int]:
    t = text.lower()
    return {k: sum(1 for p in pats if re.search(p, t)) for k, pats in TOPICS.items() if any(re.search(p, t) for p in pats)}


def relevance_score(text: str) -> tuple[float, list[str]]:
    hits = classify_topics(text)
    if not hits:
        return 0.0, []
    raw = sum(TOPIC_WEIGHT[k] * min(n, 3) for k, n in hits.items())
    return float(1 - np.exp(-raw / 1.5)), sorted(hits, key=lambda k: -hits[k])


# ---------------------------------------------------------------------- sentiment
POS = {"gain", "gains", "rally", "rallies", "surge", "surges", "soar", "record high", "beat", "beats", "strong",
       "robust", "upgrade", "optimism", "optimistic", "recovery", "rebound", "growth", "expand", "expands",
       "bullish", "outperform", "eases", "easing", "cooling inflation", "soft landing", "resilient", "boost",
       "improve", "improves", "improved", "higher profit", "upbeat", "accelerate", "stimulus", "rate cut", "cuts rates"}
NEG = {"loss", "losses", "fall", "falls", "plunge", "plunges", "slump", "selloff", "sell-off", "crash", "miss",
       "misses", "weak", "weaker", "downgrade", "fear", "fears", "pessimism", "recession", "contraction", "bearish",
       "underperform", "default", "crisis", "turmoil", "warning", "warns", "slowdown", "layoffs", "tumble",
       "tumbles", "volatile", "uncertainty", "hawkish", "rate hike", "hikes rates", "tariffs", "war", "sanctions",
       "bankruptcy", "stagflation", "inverted", "shock"}
UNCERTAIN = {"may", "might", "could", "uncertain", "unclear", "mixed", "awaits", "ahead of", "weighs", "debate",
             "volatility", "risk", "risks"}
NEGATORS = {"not", "no", "never", "without", "despite", "fails to", "failed to", "less"}
# phrases where the *macro direction* matters more than the word polarity
DIRECTIONAL_RULES = [
    (r"inflation (rises|rose|accelerat|jump|surge|hotter|higher)", -1.0),
    (r"inflation (falls|fell|cool|eas|slow|lower|declin)", +1.0),
    (r"(cpi|pce).{0,30}(above|hotter than|beat) (expectations|forecast|estimates)", -1.0),
    (r"(cpi|pce).{0,30}(below|cooler than) (expectations|forecast|estimates)", +1.0),
    (r"yields? (jump|surge|soar|spike|climb|rise)", -0.6),
    (r"yields? (fall|drop|ease|decline|slide)", +0.4),
    (r"unemployment (rises|rose|jump|climb|higher)", -0.7),
    (r"(jobless claims|layoffs) (jump|surge|rise)", -0.6),
    (r"(fed|ecb|central bank).{0,40}(cut|lower)s? (rates|interest)", +0.6),
    (r"(fed|ecb|central bank).{0,40}(hike|raise)s? (rates|interest)", -0.6),
    (r"(vix|volatility) (spike|surge|jump|soar)", -0.8),
    (r"earnings.{0,30}(beat|top|exceed)", +0.7),
    (r"earnings.{0,30}(miss|disappoint)", -0.7),
]


def lexicon_sentiment(text: str) -> tuple[float, float]:
    """Return (sentiment in [-1, 1], uncertainty in [0, 1])."""
    t = " " + text.lower() + " "
    score, n = 0.0, 0
    for rx, w in DIRECTIONAL_RULES:
        if re.search(rx, t):
            score += 2 * w
            n += 2
    tokens = re.findall(r"[a-z&\-]+", t)
    joined = " ".join(tokens)
    for lex, sign in ((POS, 1.0), (NEG, -1.0)):
        for w in lex:
            for m in re.finditer(r"\b" + re.escape(w) + r"\b", joined):
                window = joined[max(0, m.start() - 25): m.start()]
                neg = any(re.search(r"\b" + re.escape(x) + r"\b", window) for x in NEGATORS)
                score += -sign if neg else sign
                n += 1
    unc = sum(1 for w in UNCERTAIN if re.search(r"\b" + re.escape(w) + r"\b", joined))
    sent = float(np.tanh(score / max(2.0, np.sqrt(n + 1)))) if n else 0.0
    return sent, float(min(1.0, unc / 4))


def label_from(sent: float, unc: float) -> str:
    if unc >= 0.5 and abs(sent) < 0.5:
        return "UNCERTAIN"
    if sent > 0.15:
        return "POSITIVE"
    if sent < -0.15:
        return "NEGATIVE"
    return "NEUTRAL"


_FINBERT = None


def get_sentiment_model():
    """Return a callable text -> (sentiment, uncertainty). FinBERT if available, else lexicon."""
    global _FINBERT
    if _FINBERT is None:
        try:  # optional heavy dependency
            from transformers import pipeline  # type: ignore

            clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)

            def _fb(text: str):
                res = {d["label"].lower(): d["score"] for d in clf(text[:512])[0]}
                s = res.get("positive", 0) - res.get("negative", 0)
                return float(s), float(res.get("neutral", 0) * 0.5)

            _FINBERT = ("finbert", _fb)
        except Exception:
            _FINBERT = ("lexicon", lexicon_sentiment)
    return _FINBERT


# ---------------------------------------------------------------------- de-duplication
def _norm_title(t: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", "", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def deduplicate(articles: list[Article], threshold: float = 0.85) -> list[Article]:
    seen: list[tuple[str, Article]] = []
    by_hash: dict[str, Article] = {}
    for a in sorted(articles, key=lambda x: source_tier(x.source, x.url)[0]):  # keep best-tier copy
        nt = _norm_title(a.title)
        h = hashlib.md5(nt.encode()).hexdigest()
        if h in by_hash:
            a.duplicate_of = by_hash[h].article_id
            continue
        dup = next((orig for t, orig in seen if SequenceMatcher(None, t, nt).ratio() >= threshold), None)
        if dup:
            a.duplicate_of = dup.article_id
            continue
        by_hash[h] = a
        seen.append((nt, a))
    return articles


# ---------------------------------------------------------------------- events
EVENT_PATTERNS = [
    ("Monetary Policy", "Fed policy decision / guidance", r"(fomc|federal reserve|fed).{0,60}(decision|holds|cut|hike|raise|guidance|minutes|statement)", ["1D", "1W", "1M"]),
    ("Monetary Policy", "ECB policy decision", r"\becb\b.{0,60}(decision|holds|cut|hike|raise|rates)", ["1D", "1W"]),
    ("Inflation", "Inflation release / surprise", r"(cpi|pce|inflation).{0,60}(report|data|rose|fell|rises|falls|expectations|accelerat|cool)", ["1D", "1W", "1M"]),
    ("Employment", "Labour-market release", r"(payrolls|jobs report|unemployment rate|jobless claims)", ["1D", "1W"]),
    ("Growth", "Growth / recession signal", r"(gdp|recession|contraction|pmi|ism manufacturing)", ["1M", "3M", "6M"]),
    ("Earnings", "Earnings shock / guidance", r"(earnings|guidance|profit warning).{0,60}(beat|miss|cut|raise|warn|record)", ["1D", "1W"]),
    ("Volatility", "Volatility spike", r"(vix|volatility).{0,30}(spike|surge|jump|soar|highest)", ["1D", "1W", "1M"]),
    ("Geopolitical", "Geopolitical risk", r"(war|invasion|missile|sanction|tariff|trade war|conflict|attack)", ["1D", "1W", "1M"]),
    ("Financial Conditions", "Credit / banking stress", r"(bank (failure|run|collapse)|credit (crunch|spread)|default|liquidity crisis)", ["1W", "1M", "3M"]),
    ("Market Drawdown", "Equity sell-off", r"(sell-?off|plunge|crash|correction|bear market|rout)", ["1D", "1W", "1M"]),
]


@dataclass
class MarketEvent:
    event_id: str
    article_id: str
    event_date: str
    category: str
    description: str
    direction: str
    magnitude: float
    confidence: float
    horizon: list[str]
    source: str
    title: str
    available_at: str

    def to_row(self) -> dict:
        d = asdict(self)
        return {"event_id": d["event_id"], "article_id": d["article_id"], "event_date": d["event_date"],
                "category": d["category"], "description": d["description"], "direction": d["direction"],
                "magnitude": d["magnitude"], "confidence": d["confidence"], "horizon": ",".join(d["horizon"]),
                "available_at": d["available_at"], "payload": {"source": d["source"], "title": d["title"]}}


def extract_events(articles: list[Article]) -> list[MarketEvent]:
    out = []
    for a in articles:
        if a.duplicate_of or a.relevance < 0.25:
            continue
        text = f"{a.title}. {a.summary}".lower()
        for cat, desc, rx, hz in EVENT_PATTERNS:
            if re.search(rx, text):
                direction = ("Potentially bullish" if a.sentiment > 0.2 else
                             "Potentially bearish" if a.sentiment < -0.2 else "Uncertain")
                magnitude = round(100 * min(1.0, abs(a.sentiment) * 0.6 + a.relevance * 0.4), 1)
                confidence = round(100 * a.source_quality * (0.5 + 0.5 * a.relevance), 1)
                eid = hashlib.sha1(f"{a.article_id}{cat}".encode()).hexdigest()[:16]
                out.append(MarketEvent(eid, a.article_id, a.published_at[:10], cat, desc, direction, magnitude,
                                       confidence, hz, a.source, a.title, a.published_at))
    return out


# ---------------------------------------------------------------------- analysis
def analyze_articles(articles: list[Article]) -> list[Article]:
    model_name, model = get_sentiment_model()
    deduplicate(articles)
    for a in articles:
        text = f"{a.title}. {a.summary}"
        a.relevance, a.topics = relevance_score(text)
        tier, quality = source_tier(a.source, a.url)
        freshness = 1.0
        a.source_quality = round(quality * freshness, 3)
        s, unc = model(text)
        a.sentiment = round(float(s), 3)
        a.label = label_from(s, unc)
    return articles


def corroboration(articles: list[Article]) -> dict[str, int]:
    """Number of distinct sources covering each topic (independent corroboration)."""
    srcs = defaultdict(set)
    for a in articles:
        if a.duplicate_of:
            continue
        for t in a.topics:
            srcs[t].add(a.source)
    return {k: len(v) for k, v in srcs.items()}


def detect_contradictions(articles: list[Article]) -> list[dict]:
    """Topics where credible sources disagree in direction."""
    by_topic = defaultdict(list)
    for a in articles:
        if a.duplicate_of or a.relevance < 0.3 or a.source_quality < 0.5:
            continue
        for t in a.topics[:2]:
            by_topic[t].append(a)
    out = []
    for t, arts in by_topic.items():
        pos = [a for a in arts if a.label == "POSITIVE"]
        neg = [a for a in arts if a.label == "NEGATIVE"]
        if pos and neg and min(len(pos), len(neg)) / max(len(pos), len(neg)) >= 0.34:
            out.append({"topic": t, "positive": len(pos), "negative": len(neg),
                        "example_positive": pos[0].title, "example_negative": neg[0].title})
    return out


def aggregate_sentiment(articles: list[Article]) -> dict:
    valid = [a for a in articles if not a.duplicate_of and a.relevance >= 0.2]
    if not valid:
        return {"score": np.nan, "dispersion": np.nan, "n": 0, "weighted_n": 0.0, "by_topic": {}, "label_counts": {}}
    w = np.array([a.relevance * a.source_quality for a in valid])
    s = np.array([a.sentiment for a in valid])
    # cap the weight any single low-tier source can contribute
    for i, a in enumerate(valid):
        if source_tier(a.source, a.url)[0] >= 4:
            w[i] = min(w[i], 0.25)
    score = float(np.average(s, weights=w)) if w.sum() > 0 else 0.0
    disp = float(np.sqrt(np.average((s - score) ** 2, weights=w))) if w.sum() > 0 else 0.0
    by_topic = {}
    for t in TOPICS:
        ts = [(a.sentiment, a.relevance * a.source_quality) for a in valid if t in a.topics]
        if ts:
            arr = np.array(ts)
            by_topic[t] = {"score": round(float(np.average(arr[:, 0], weights=arr[:, 1] + 1e-9)), 3), "n": len(ts)}
    labels = pd.Series([a.label for a in valid]).value_counts().to_dict()
    return {"score": round(score, 3), "dispersion": round(disp, 3), "n": len(valid), "weighted_n": round(float(w.sum()), 2),
            "by_topic": by_topic, "label_counts": labels, "model": get_sentiment_model()[0]}
