"""Provider abstraction, fallback chains and source-health tracking.

Every provider call returns a :class:`FetchResult` describing *what happened*
(success, latency, rows, error). Failures are never hidden: the fallback chain records
each attempt so the Data Quality tab can display source health.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from ...core.runtime import get_logger, utcnow_iso

log = get_logger("vusa.sources")


class ProviderUnavailable(RuntimeError):
    """Raised when a provider cannot serve a request (missing key, network, empty data)."""


@dataclass
class FetchResult:
    dataset: str
    source: str
    ok: bool
    data: Any = None
    rows: int = 0
    latency_ms: float = 0.0
    error: str = ""
    retrieved_at: str = field(default_factory=utcnow_iso)
    last_date: str | None = None
    is_synthetic: bool = False
    from_cache: bool = False

    def health_row(self) -> dict:
        status = "ok" if self.ok else "failed"
        if self.ok and self.from_cache:
            status = "cached"
        if self.ok and self.is_synthetic:
            status = "SYNTHETIC DEMO"
        return {"dataset": self.dataset, "source": self.source, "status": status, "latency_ms": round(self.latency_ms, 1),
                "rows": self.rows, "last_date": self.last_date, "issues": self.error, "checked_at": self.retrieved_at}


class SourceHealth:
    """In-memory registry of the latest health of every (dataset, source) pair."""

    def __init__(self):
        self.records: dict[tuple[str, str], FetchResult] = {}

    def record(self, res: FetchResult) -> None:
        self.records[(res.dataset, res.source)] = res

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([r.health_row() for r in self.records.values()])

    def any_ok(self, dataset: str) -> bool:
        return any(r.ok for (d, _), r in self.records.items() if d == dataset)


def run_with_fallback(dataset: str, attempts: list[tuple[str, Callable[[], Any]]], health: SourceHealth | None = None,
                      validate: Callable[[Any], None] | None = None) -> tuple[FetchResult, list[FetchResult]]:
    """Try providers in priority order: SOURCE A -> SOURCE B -> SOURCE C.

    Returns the first successful result plus the full attempt log.
    """
    log_attempts: list[FetchResult] = []
    for name, fn in attempts:
        t0 = time.perf_counter()
        try:
            data = fn()
            if data is None or (hasattr(data, "empty") and data.empty):
                raise ProviderUnavailable("provider returned no data")
            if validate:
                validate(data)
            res = FetchResult(dataset, name, True, data, rows=len(data) if hasattr(data, "__len__") else 1,
                              latency_ms=(time.perf_counter() - t0) * 1000)
            if isinstance(data, (pd.DataFrame, pd.Series)) and len(data):
                res.last_date = str(pd.Timestamp(data.index[-1]).date())
            log_attempts.append(res)
            if health:
                health.record(res)
            return res, log_attempts
        except Exception as exc:  # noqa: BLE001 - provider errors are reported, not swallowed
            res = FetchResult(dataset, name, False, error=f"{type(exc).__name__}: {exc}"[:400],
                              latency_ms=(time.perf_counter() - t0) * 1000)
            log.warning("%s via %s failed: %s", dataset, name, res.error)
            log_attempts.append(res)
            if health:
                health.record(res)
    return FetchResult(dataset, "none", False, error="all providers failed: " +
                       "; ".join(f"{a.source}: {a.error}" for a in log_attempts)), log_attempts


def http_get(url: str, timeout: float = 20.0, params: dict | None = None, headers: dict | None = None):
    import requests

    h = {"User-Agent": "VUSA-Quant-Research/1.0 (personal research; respects robots/ToS)"}
    h.update(headers or {})
    r = requests.get(url, params=params, timeout=timeout, headers=h)
    r.raise_for_status()
    return r
