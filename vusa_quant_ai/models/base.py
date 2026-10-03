"""Common interface for every forecasting model.

A model produces, for each row of X (one decision time), an expected forward return for
its horizon and a probability that the return is positive. Preprocessing (imputation,
scaling) is fitted inside ``fit`` only, so it can never see test data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class Preprocessor:
    """Median imputation + robust scaling, fitted on training rows only."""

    def __init__(self, clip: float = 8.0):
        self.clip = clip
        self.med_: pd.Series | None = None
        self.scale_: pd.Series | None = None
        self.cols_: list[str] = []
        self.fit_last_index_ = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.cols_ = list(X.columns)
        self.med_ = X.median()
        q75, q25 = X.quantile(0.75), X.quantile(0.25)
        self.scale_ = ((q75 - q25) / 1.349).replace(0, np.nan).fillna(X.std()).replace(0, 1.0).fillna(1.0)
        self.fit_last_index_ = X.index[-1] if len(X) else None
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.cols_].fillna(self.med_).fillna(0.0)
        Z = (Z - self.med_.fillna(0.0)) / self.scale_
        return np.clip(Z.values.astype(np.float32), -self.clip, self.clip)


@dataclass
class ModelInfo:
    name: str
    family: str  # statistical | ml | deep_learning | baseline
    description: str
    hyperparameters: dict = field(default_factory=dict)
    available: bool = True
    unavailable_reason: str = ""


class BaseForecaster:
    info: ModelInfo
    needs_context = False  # sequence / time-series models need the full history

    def __init__(self, horizon: int, seed: int = 42, **hp):
        self.horizon = horizon
        self.seed = seed
        self.hp = hp
        self.context_X: pd.DataFrame | None = None
        self.context_close: pd.Series | None = None
        self.resid_std_: float = np.nan

    def set_context(self, X_full: pd.DataFrame | None, close: pd.Series | None) -> None:
        self.context_X, self.context_close = X_full, close

    def fit(self, X: pd.DataFrame, y_ret: pd.Series, y_up: pd.Series, w: np.ndarray | None = None) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame(index=X.index, columns=['ret', 'p_up'])."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.info.name


def normal_p_up(mu: np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    sigma = np.where(np.asarray(sigma) > 0, sigma, np.nan)
    return np.clip(norm.cdf(np.asarray(mu) / sigma), 0.01, 0.99)


class NaiveDrift(BaseForecaster):
    """Benchmark: historical mean return and historical frequency of positive returns.

    Every other model must beat this (it encodes the equity risk premium only)."""

    info = ModelInfo("naive_drift", "baseline", "Historical mean drift / base-rate benchmark")

    def fit(self, X, y_ret, y_up, w=None):
        w = np.ones(len(y_ret)) if w is None else w
        m = y_ret.notna().values
        self.mu_ = float(np.average(y_ret[m], weights=w[m]))
        self.p_ = float(np.average(y_up[m], weights=w[m]))
        self.resid_std_ = float(y_ret[m].std())
        return self

    def predict(self, X):
        return pd.DataFrame({"ret": self.mu_, "p_up": self.p_}, index=X.index)
