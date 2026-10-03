"""LEAKAGE AUDIT.

Run before any backtest / validation result is accepted. Each check returns PASS, WARN
or FAIL with an exact explanation. Any FAIL stops the backtest.

Checks
------
1. future_dates          - feature rows dated after the as-of date
2. future_price_leakage  - a feature is (nearly) identical to a forward return
3. target_contamination  - a feature correlates implausibly with the label in-sample
4. overlapping_windows   - train/test label windows overlap (purging verified)
5. scaler_leakage        - preprocessing fitted on data beyond the training window
6. feature_shift_test    - features change when *future* rows are removed (non-causal computation)
7. news_leakage          - news used with publication time after the decision time
8. macro_release_leakage - macro value used before its publication time
9. revision_risk         - macro series whose history is latest-vintage (WARN, not FAIL)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AuditCheck:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


@dataclass
class LeakageAudit:
    checks: list[AuditCheck] = field(default_factory=list)

    def add(self, name, status, detail):
        self.checks.append(AuditCheck(name, status, detail))

    @property
    def passed(self) -> bool:
        return not any(c.status == "FAIL" for c in self.checks)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [c.__dict__ for c in self.checks]}

    def summary(self) -> str:
        head = "LEAKAGE AUDIT PASSED" if self.passed else "LEAKAGE AUDIT FAILED - BACKTEST STOPPED"
        return head + "\n" + "\n".join(f"  [{c.status}] {c.name}: {c.detail}" for c in self.checks)


class LeakageError(RuntimeError):
    def __init__(self, audit: LeakageAudit):
        super().__init__(audit.summary())
        self.audit = audit


def check_future_dates(X: pd.DataFrame, as_of) -> AuditCheck:
    bad = X.index > pd.Timestamp(as_of)
    if bad.any():
        return AuditCheck("future_dates", "FAIL", f"{bad.sum()} feature rows dated after as-of {pd.Timestamp(as_of).date()}")
    return AuditCheck("future_dates", "PASS", "no feature rows after the as-of date")


def check_future_price_leakage(X: pd.DataFrame, close: pd.Series, horizons=(1, 5, 20)) -> AuditCheck:
    offenders = []
    for h in horizons:
        fwd = close.shift(-h) / close - 1
        for c in X.columns:
            a = pd.concat([X[c], fwd], axis=1).dropna()
            if len(a) > 100:
                corr = a.iloc[:, 0].corr(a.iloc[:, 1])
                if abs(corr) > 0.97:
                    offenders.append(f"{c}~fwd{h}d (r={corr:.3f})")
    if offenders:
        return AuditCheck("future_price_leakage", "FAIL", "features replicate forward returns: " + ", ".join(offenders[:5]))
    return AuditCheck("future_price_leakage", "PASS", "no feature replicates a forward return (|r|<=0.97)")


def check_target_contamination(X: pd.DataFrame, y: pd.Series, max_abs_corr: float = 0.6) -> AuditCheck:
    a = X.join(y.rename("__y"), how="inner").dropna(subset=["__y"])
    sus = []
    for c in X.columns:
        s = a[[c, "__y"]].dropna()
        if len(s) > 200:
            r = s[c].corr(s["__y"], method="spearman")
            if abs(r) > max_abs_corr:
                sus.append(f"{c} (rho={r:.2f})")
    if sus:
        return AuditCheck("target_contamination", "FAIL",
                          "implausibly strong in-sample feature/label correlation - likely uses label information: "
                          + ", ".join(sus[:5]))
    return AuditCheck("target_contamination", "PASS", f"max |rank corr| with target below {max_abs_corr}")


def check_overlap(splits, index: pd.DatetimeIndex, label_end: pd.Series) -> AuditCheck:
    for s in splits:
        t0 = index[s.test[0]]
        tr_end = label_end.iloc[s.train]
        before = index[s.train] < t0
        if (tr_end[before] >= t0).any():
            n = int((tr_end[before] >= t0).sum())
            return AuditCheck("overlapping_windows", "FAIL",
                              f"fold {s.fold}: {n} training labels end inside the test window (purging failed)")
        if len(np.intersect1d(s.train, s.test)):
            return AuditCheck("overlapping_windows", "FAIL", f"fold {s.fold}: train and test share rows")
    return AuditCheck("overlapping_windows", "PASS", "label windows purged; no train/test overlap")


def check_scaler(fit_ranges: list[tuple], splits) -> AuditCheck:
    """fit_ranges: list of (fold, last_train_position_used_for_fitting_preprocessing)."""
    for (fold, last_pos), s in zip(fit_ranges, splits):
        if last_pos > s.train.max():
            return AuditCheck("scaler_leakage", "FAIL", f"fold {fold}: preprocessing fitted on rows beyond training window")
    return AuditCheck("scaler_leakage", "PASS", "scalers / imputers fitted inside each training fold only")


def check_feature_causality(build_fn, data: pd.DataFrame, cut_points: list[int], tol: float = 1e-8,
                            sample_cols: int = 400) -> AuditCheck:
    """Recompute features on data truncated at a cut point; values at the cut must not change."""
    full = build_fn(data)
    bad = []
    for cp in cut_points:
        part = build_fn(data.iloc[: cp + 1])
        d = data.index[cp]
        cols = [c for c in full.columns if c in part.columns][:sample_cols]
        a, b = full.loc[d, cols].astype(float), part.loc[d, cols].astype(float)
        diff = (a - b).abs()
        rel = diff / (a.abs() + 1e-9)
        changed = [c for c in cols if not (pd.isna(a[c]) and pd.isna(b[c])) and
                   (pd.isna(a[c]) != pd.isna(b[c]) or (diff[c] > tol and rel[c] > 1e-6))]
        if changed:
            bad.append(f"{d.date()}: {', '.join(changed[:6])}")
    if bad:
        return AuditCheck("feature_shift_test", "FAIL", "features depend on future rows -> " + " | ".join(bad[:3]))
    return AuditCheck("feature_shift_test", "PASS", f"features unchanged when future rows removed ({len(cut_points)} cut points)")


def check_news_leakage(news_used: pd.DataFrame, decision_time) -> AuditCheck:
    if news_used is None or news_used.empty:
        return AuditCheck("news_leakage", "PASS", "no news inputs used")
    pub = pd.to_datetime(news_used["published_at"], utc=True)
    late = pub > pd.Timestamp(decision_time).tz_convert("UTC") if pd.Timestamp(decision_time).tzinfo else \
        pub > pd.Timestamp(decision_time, tz="UTC")
    if late.any():
        return AuditCheck("news_leakage", "FAIL", f"{late.sum()} articles published after decision time were used")
    return AuditCheck("news_leakage", "PASS", "all news published before decision time")


def check_macro_release(pit, decision_times: pd.DatetimeIndex, sample: int = 40) -> list[AuditCheck]:
    out = []
    if not pit.series_ids:
        return [AuditCheck("macro_release_leakage", "PASS", "no macro inputs")]
    rng = np.random.default_rng(0)
    pos = rng.choice(len(decision_times), min(sample, len(decision_times)), replace=False)
    viol = []
    latest_vintage = []
    for sid in pit.series_ids:
        for p in pos:
            t = decision_times[p]
            _, eff, pub = pit.value_as_of(sid, t)
            if pub is not None and pub > t:
                viol.append(f"{sid}@{t.date()}")
        if pit.method(sid) == "release_lag":
            latest_vintage.append(sid)
    out.append(AuditCheck("macro_release_leakage", "FAIL" if viol else "PASS",
                          ("macro values used before publication: " + ", ".join(viol[:5])) if viol else
                          "every macro value was published before the decision time"))
    if latest_vintage:
        out.append(AuditCheck("revision_risk", "WARN",
                              f"{len(latest_vintage)} macro series use latest-vintage data with release-lag timing "
                              f"({', '.join(latest_vintage[:6])}{'...' if len(latest_vintage) > 6 else ''}); later "
                              "revisions may leak. Add a FRED API key for true ALFRED vintages."))
    else:
        out.append(AuditCheck("revision_risk", "PASS", "macro history uses true real-time vintages"))
    return out


def run_audit(X: pd.DataFrame, y: pd.Series, close: pd.Series, as_of, splits=None, index=None, label_end=None,
              pit=None, decision_times=None, fit_ranges=None, causality_fn=None, causality_data=None,
              news_used=None, decision_time=None) -> LeakageAudit:
    audit = LeakageAudit()
    audit.checks.append(check_future_dates(X, as_of))
    audit.checks.append(check_future_price_leakage(X, close.reindex(X.index)))
    audit.checks.append(check_target_contamination(X, y))
    if splits is not None:
        audit.checks.append(check_overlap(splits, index, label_end))
    if fit_ranges is not None and splits is not None:
        audit.checks.append(check_scaler(fit_ranges, splits))
    if causality_fn is not None and causality_data is not None:
        n = len(causality_data)
        cps = [int(n * q) for q in (0.5, 0.8, 0.95) if int(n * q) > 300]
        if cps:
            audit.checks.append(check_feature_causality(causality_fn, causality_data, cps))
    if pit is not None and decision_times is not None:
        audit.checks.extend(check_macro_release(pit, decision_times))
    if news_used is not None or decision_time is not None:
        audit.checks.append(check_news_leakage(news_used, decision_time or pd.Timestamp.now(tz="UTC")))
    return audit
