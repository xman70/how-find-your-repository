"""Statistical time-series forecasters: ARIMA, ETS (damped Holt), structural state-space
(local linear trend, a frequentist analogue of Bayesian structural time series), and a
seasonal SARIMA variant.

Parameters are estimated on the training window only. Forecasts for later origins use
the *fixed* parameters and the causal Kalman / exponential-smoothing filter, so the state
at origin t depends only on data up to t. The h-step cumulative log-return forecast is
computed by propagating the filtered state through the transition matrix.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..base import BaseForecaster, ModelInfo, normal_p_up

warnings.filterwarnings("ignore", message=".*frequency information.*")
warnings.filterwarnings("ignore", message=".*Non-stationary starting.*")
warnings.filterwarnings("ignore", message=".*Non-invertible starting.*")


def _origins_positions(close: pd.Series, X: pd.DataFrame) -> np.ndarray:
    pos = close.index.get_indexer(X.index)
    if (pos < 0).any():
        raise ValueError("prediction rows not found in context series")
    return pos


class StateSpaceForecaster(BaseForecaster):
    needs_context = True
    max_train = 2520  # estimate on at most the last ~10 years (speed / stability)

    def _build(self, endog):
        raise NotImplementedError

    def _endog(self, close: pd.Series) -> pd.Series:
        return np.log(close).diff().dropna() * 100  # daily % log returns

    def fit(self, X, y_ret, y_up, w=None):
        close = self.context_close
        end = X.index[-1]
        series = self._endog(close.loc[:end]).iloc[-self.max_train:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = self._build(series)
            self.res_ = mod.fit(disp=False, maxiter=200)
        # aleatoric scale: dispersion of realised h-sums vs forecasts on training rows
        tr = self._forecast_positions(close.loc[:end], X.index[-min(len(X), 750):])
        yy = y_ret.reindex(tr.index)
        r = np.log1p(yy) - tr
        self.resid_std_ = float(np.nanstd(r)) if r.notna().sum() > 30 else float(np.nanstd(np.log1p(y_ret)))
        self.train_end_ = end
        return self

    def _forecast_positions(self, close: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
        series = self._endog(close)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self.res_.apply(series)
        ssm = applied.model.ssm
        T = np.asarray(ssm["transition"])[:, :, 0] if ssm["transition"].ndim == 3 else np.asarray(ssm["transition"])
        Z = np.asarray(ssm["design"])[:, :, 0] if ssm["design"].ndim == 3 else np.asarray(ssm["design"])
        c = np.asarray(ssm["state_intercept"])[:, 0] if ssm["state_intercept"].ndim == 2 else np.asarray(ssm["state_intercept"])
        d = float(np.ravel(ssm["obs_intercept"])[0]) if np.size(ssm["obs_intercept"]) else 0.0
        A = np.asarray(applied.filtered_state)  # (m, n) state at t given data up to t
        pos = series.index.get_indexer(idx)
        ok = pos >= 0
        A = A[:, pos[ok]]
        total = np.zeros(A.shape[1])
        for _ in range(self.horizon):
            A = T @ A + c[:, None]
            total += (Z @ A).ravel() + d
        out = pd.Series(np.nan, index=idx)
        out[ok] = total / 100.0  # back to log-return units
        return out

    def predict(self, X):
        close = self.context_close.loc[: X.index[-1]]
        lr = self._forecast_positions(close, X.index).values
        sigma = self.resid_std_ if np.isfinite(self.resid_std_) else np.nan
        return pd.DataFrame({"ret": np.expm1(lr), "p_up": normal_p_up(lr, sigma)}, index=X.index)


class ARIMAForecaster(StateSpaceForecaster):
    info = ModelInfo("arima", "statistical", "ARIMA(p,0,q) on daily log returns (state-space)", {"order": (2, 0, 1)})

    def _build(self, endog):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX(endog, order=tuple(self.hp.get("order", (2, 0, 1))), trend="c")


class SARIMAForecaster(StateSpaceForecaster):
    info = ModelInfo("sarima", "statistical", "SARIMA with weekly (5-session) seasonality",
                     {"order": (1, 0, 1), "seasonal_order": (1, 0, 0, 5)})

    def _build(self, endog):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX(endog, order=(1, 0, 1), seasonal_order=(1, 0, 0, 5), trend="c")


class StructuralForecaster(StateSpaceForecaster):
    info = ModelInfo("state_space", "statistical",
                     "Structural time-series (local level + stochastic drift) on log price - BSTS analogue")
    max_train = 1500

    def _endog(self, close):
        return np.log(close.dropna()) * 100

    def _build(self, endog):
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        return UnobservedComponents(endog, level="local linear trend")

    def _forecast_positions(self, close, idx):
        # forecast of log price at t+h minus current filtered level -> cumulative log return
        series = self._endog(close)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self.res_.apply(series)
        A = np.asarray(applied.filtered_state)
        pos = series.index.get_indexer(idx)
        ok = pos >= 0
        lvl, slope = A[0, pos[ok]], A[1, pos[ok]]
        obs = series.values[pos[ok]]
        out = pd.Series(np.nan, index=idx)
        out[ok] = (lvl + self.horizon * slope - obs) / 100.0
        return out


class ETSForecaster(BaseForecaster):
    """Damped-trend Holt exponential smoothing on log price (parameters from statsmodels)."""

    info = ModelInfo("ets", "statistical", "ETS(A,Ad,N) damped-trend exponential smoothing on log price")
    needs_context = True

    def fit(self, X, y_ret, y_up, w=None):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        end = X.index[-1]
        lp = np.log(self.context_close.loc[:end].dropna()).iloc[-2520:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ExponentialSmoothing(lp.values, trend="add", damped_trend=True).fit()
        p = res.params
        self.alpha_, self.beta_ = float(p["smoothing_level"]), float(p["smoothing_trend"])
        self.beta_ = float(np.clip(self.beta_, 0, 1))
        self.phi_ = float(p.get("damping_trend", 0.98))
        if not np.isfinite(self.phi_):
            self.phi_ = 0.98
        tr = self._filter(self.context_close.loc[:end], X.index[-min(len(X), 750):])
        r = np.log1p(y_ret.reindex(tr.index)) - tr
        self.resid_std_ = float(np.nanstd(r))
        return self

    def _filter(self, close, idx):
        lp = np.log(close.dropna())
        y = lp.values
        l, b = y[0], 0.0
        fc = np.empty(len(y))
        damp = sum(self.phi_ ** i for i in range(1, self.horizon + 1))
        for t in range(len(y)):
            l_prev = l
            l = self.alpha_ * y[t] + (1 - self.alpha_) * (l + self.phi_ * b)
            b = self.beta_ * (l - l_prev) + (1 - self.beta_) * self.phi_ * b
            fc[t] = l + damp * b - y[t]
        s = pd.Series(fc, index=lp.index)
        return s.reindex(idx)

    def predict(self, X):
        lr = self._filter(self.context_close.loc[: X.index[-1]], X.index).values
        return pd.DataFrame({"ret": np.expm1(lr), "p_up": normal_p_up(lr, self.resid_std_)}, index=X.index)


STAT_MODELS = {c.info.name: c for c in (ARIMAForecaster, SARIMAForecaster, StructuralForecaster, ETSForecaster)}
