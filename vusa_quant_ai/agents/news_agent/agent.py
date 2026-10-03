"""Agent 3 - News: current news intelligence (MARKET RESEARCH AGENT).

Daily steps: search/collect -> de-duplicate -> relevance -> event extraction ->
POSITIVE/NEGATIVE/NEUTRAL/UNCERTAIN classification -> relevance & source reliability ->
contradiction detection -> structured market-intelligence report.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ...data.news.ingest import collect_news
from ...data.news.intelligence import (aggregate_sentiment, analyze_articles, corroboration, detect_contradictions,
                                       extract_events, source_tier)
from ..base import Agent, Finding


class MarketResearchAgent:
    """Runs the 10-step daily research loop and returns a structured report."""

    def __init__(self, settings, health=None):
        self.s, self.health = settings, health

    def run(self, offline: bool = False) -> dict:
        steps = []
        if offline:
            return {"available": False, "reason": "offline mode - no current news retrieved", "articles": [], "events": [],
                    "sentiment": {}, "contradictions": [], "steps": ["skipped (offline)"], "sources": []}
        arts, attempts = collect_news(self.s.news.rss_feeds, self.s.api_keys.newsapi, self.s.providers.news_priority,
                                      self.s.news.lookback_days, self.s.news.max_articles, self.s.providers.request_timeout,
                                      self.health)
        steps.append(f"1-2 search & collect: {len(arts)} articles from {sum(a.ok for a in attempts)}/{len(attempts)} sources")
        if not arts:
            return {"available": False, "reason": "no news source reachable: " + "; ".join(
                f"{a.source}: {a.error[:80]}" for a in attempts if not a.ok)[:600], "articles": [], "events": [],
                "sentiment": {}, "contradictions": [], "steps": steps,
                "sources": [a.health_row() for a in attempts]}
        analyze_articles(arts)
        n_dup = sum(1 for a in arts if a.duplicate_of)
        steps.append(f"3 de-duplicate: {n_dup} duplicates removed")
        rel = [a for a in arts if not a.duplicate_of and a.relevance >= 0.25]
        steps.append(f"4 relevance: {len(rel)} relevant to S&P 500 / macro / policy")
        events = extract_events(arts)
        steps.append(f"5 events extracted: {len(events)}")
        steps.append("6-8 classified, relevance and source reliability scored: " +
                     ", ".join(f"{k}={v}" for k, v in Counter(a.label for a in rel).items()))
        contra = detect_contradictions(arts)
        steps.append(f"9 contradictions: {len(contra)}")
        sent = aggregate_sentiment(arts)
        steps.append("10 report generated")
        tiers = Counter(source_tier(a.source, a.url)[0] for a in rel)
        return {"available": True, "articles": [a.to_row() for a in arts], "events": [e.to_row() | {"title": e.title,
                "source": e.source, "horizon_list": e.horizon} for e in events], "sentiment": sent,
                "contradictions": contra, "corroboration": corroboration(arts), "steps": steps,
                "source_tiers": dict(tiers), "sources": [a.health_row() for a in attempts],
                "news_shock": bool(sum(1 for e in events if e.direction == "Potentially bearish" and e.magnitude > 60
                                       and e.confidence > 60) >= 3)}


class NewsAgent(Agent):
    name = "News"
    role = "Current news and event intelligence"

    def run(self, state: dict):
        news = state.get("news", {})
        if not news.get("available"):
            return self._out([], [], [f"News unavailable: {news.get('reason', 'not run')}"], score=0.0)
        f, concerns = [], []
        s = news.get("sentiment", {})
        if s.get("n", 0):
            f.append(Finding(f"Quality-weighted news sentiment {s['score']:+.2f} over {s['n']} relevant articles "
                             f"(dispersion {s['dispersion']:.2f}, model: {s.get('model')})",
                             "bullish" if s["score"] > 0.15 else "bearish" if s["score"] < -0.15 else "neutral", 1.0))
        for t, v in sorted(s.get("by_topic", {}).items(), key=lambda kv: -kv[1]["n"])[:4]:
            f.append(Finding(f"Topic {t}: sentiment {v['score']:+.2f} (n={v['n']})",
                             "bullish" if v["score"] > 0.2 else "bearish" if v["score"] < -0.2 else "neutral", 0.4))
        evs = news.get("events", [])
        big = sorted(evs, key=lambda e: -(e["magnitude"] * e["confidence"]))[:5]
        for e in big:
            d = "bullish" if e["direction"] == "Potentially bullish" else "bearish" if e["direction"] == "Potentially bearish" else "neutral"
            f.append(Finding(f"[{e['category']}] {e['title'][:110]} ({e['source']}; magnitude {e['magnitude']:.0f}, "
                             f"confidence {e['confidence']:.0f})", d, 0.5))
        for c in news.get("contradictions", []):
            concerns.append(f"Contradictory coverage on {c['topic']}: {c['positive']} positive vs {c['negative']} negative")
        for m in state.get("news_memory", [])[:4]:
            if m.get("20D_mean") is not None and np.isfinite(m["20D_mean"]):
                f.append(Finding(f"News memory: after past '{m['historical_category']}' events (n={m['n']}) the 20D mean "
                                 f"return was {m['20D_mean']:+.1%} vs base {m.get('20D_base_mean', np.nan):+.1%}", "neutral", 0.3))
        if news.get("news_shock"):
            concerns.append("Several high-magnitude, high-confidence bearish events: NEWS SHOCK.")
        return self._out(f, concerns)
