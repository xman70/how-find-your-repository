"""ForecastService - multi-horizon, multi-model, probabilistic forecasting.

For every horizon:

1. Labels (executable next-open forward return, direction, drawdown, triple barrier)
2. Purged / embargoed expanding walk-forward folds
3. Inside each fold: feature selection on the training window only, then every base
   model is fitted and predicts the untouched test window -> out-of-sample (OOS) matrix
4. Ensemble methods are compared on the OOS matrix with their own expanding,
   non-overlapping estimation (single best / equal / performance / stacked / Bayesian)
5. Probability calibration and conformal intervals are fitted on OOS outputs only
6. Final refit on all *labelled* history (rows whose label window ended before the
   as-of date) and prediction for the as-of row
7. Uncertainty decomposition: aleatoric (residual dispersion) vs epistemic (model
   disagreement); disagreement reduces confidence

Nothing is reported that was not computed. Models that fail to train are listed with the
error instead of being dropped silently.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..config.settings import Settings
from ..core.runtime import get_logger, set_seeds
from ..features.builder import FeatureSet
from ..features.selection import SelectionTracker, select_features
from ..labeling.labels import build_labels
from ..validation.leakage import LeakageAudit, run_audit
from ..validation.metrics import brier, classification_metrics, diebold_mariano, interval_metrics, regression_metrics
from ..validation.splits import sample_weights, walk_forward
from .calibration.calibration import ConformalRegressor, ProbabilityCalibrator, calibration_report, rolling_conformal_backtest
from .ensemble.ensemble import METHODS, combine, walk_forward_ensembles
from .registry import ALL_MODELS, ModelRegistry

log = get_logger("vusa.forecast")

REGIME_FEATURES = ["regime_score", "hmm_stress_prob", "rv_pct"]


@dataclass
class HorizonResult:
    horizon: int
    as_of: str
    expected_return: float = np.nan
    p_up_raw: float = np.nan
    p_up: float = np.nan
    p_down: float = np.nan
    p_gt5: float = np.nan
    p_lt5: float = np.nan
    p_dd10: float = np.nan
    interval: tuple = (np.nan, np.nan)
    interval_level: float = 0.8
    quantiles: dict = field(default_factory=dict)
    model_preds: dict = field(default_factory=dict)  # name -> {"ret", "p_up"}
    model_errors: dict = field(default_factory=dict)
    ensemble_method: str = "equal_weight"
    ensemble_weights: dict = field(default_factory=dict)
    ensemble_comparison: dict = field(default_factory=dict)
    model_scores: dict = field(default_factory=dict)
    scores_by_regime: dict = field(default_factory=dict)
    dm_tests: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    calibration_curve: list = field(default_factory=list)
    conformal: dict = field(default_factory=dict)
    disagreement: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)
    confidence: float = np.nan
    selected_features: list = field(default_factory=list)
    feature_scores: pd.DataFrame | None = None
    oos: pd.DataFrame | None = None  # OOS predictions (ensemble + base) with truth
    n_train: int = 0
    n_oos: int = 0
    model_ids: dict = field(default_factory=dict)
    folds: list = field(default_factory=list)
    final_models: dict = field(default_factory=dict)
    hpo: dict = field(default_factory=dict)  # model -> list of per-fold best params (nested)
    warnings: list = field(default_factory=list)


@dataclass
class ForecastResult:
    as_of: str
    horizons: dict[int, HorizonResult]
    audit: LeakageAudit | None
    selection: pd.DataFrame
    training_window: str
    window_comparison: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    hpo: dict = field(default_factory=dict)


def _composite(m: dict) -> float:
    """Robust model-selection score: rank IC + directional skill - calibration error (higher is better)."""
    ic = m.get("ic_spearman", 0) or 0
    da = (m.get("directional_accuracy", 0.5) or 0.5) - 0.5
    b = m.get("brier", 0.25) or 0.25
    return ic + da - 2 * (b - 0.25)


class ForecastService:
    def __init__(self, settings: Settings, registry: ModelRegistry | None = None,
                 progress: Callable[[str, float], None] | None = None, hpo_trials: int | None = None):
        self.s = settings
        self.hpo_trials = settings.models.optuna_trials if hpo_trials is None else hpo_trials
        self.registry = registry
        self.progress = progress or (lambda msg, frac: None)

    # ------------------------------------------------------------------ helpers
    def _make(self, name: str, h: int, hp: dict | None = None):
        cls = ALL_MODELS[name]
        kw = dict(hp or {})
        if cls.info.family == "deep_learning":
            kw["epochs"] = self.s.models.deep_learning_epochs
        return cls(h, seed=self.s.models.random_seed, **kw)

    def available_models(self, names: list[str]) -> tuple[list[str], dict[str, str]]:
        ok, skipped = [], {}
        for n in names:
            if n not in ALL_MODELS:
                skipped[n] = "unknown model"
            elif not ALL_MODELS[n].info.available:
                skipped[n] = ALL_MODELS[n].info.unavailable_reason
            else:
                ok.append(n)
        if "naive_drift" not in ok:
            ok.insert(0, "naive_drift")  # benchmark is always present
        return ok, skipped

    @staticmethod
    def design_matrix(fs: FeatureSet, regimes: pd.DataFrame | None, exclude_groups=()) -> pd.DataFrame:
        X = fs.frame[fs.model_columns(exclude_groups)].copy()
        if regimes is not None:
            for c in REGIME_FEATURES:
                if c in regimes:
                    X[c] = regimes[c].reindex(X.index).astype(float)
        return X

    # ------------------------------------------------------------------ main
    def run(self, fs: FeatureSet, target: pd.DataFrame, regimes: pd.DataFrame | None, horizons: list[int],
            model_names: list[str], as_of=None, pit=None, full_validation: bool = False,
            exclude_groups: tuple[str, ...] = (), register: bool = True) -> ForecastResult:
        set_seeds(self.s.models.random_seed)
        t_start = time.perf_counter()
        as_of = pd.Timestamp(as_of or target.index[-1])
        X_all = self.design_matrix(fs, regimes, exclude_groups).loc[:as_of]
        tgt = target.loc[:as_of]
        labels = build_labels(tgt, horizons)
        models, skipped = self.available_models(model_names)
        tracker = SelectionTracker()
        primary = self.s.models.primary_horizon if self.s.models.primary_horizon in horizons else horizons[len(horizons) // 2]

        # --------------------------------------------------- leakage audit (blocking)
        lab_p = labels[primary]
        audit = run_audit(X_all.iloc[:-primary - 1], lab_p["fwd_ret"].iloc[:-primary - 1], tgt["close"], as_of,
                          pit=pit, decision_times=fs.decision_times[: len(X_all)] if pit is not None else None,
                          causality_fn=(lambda d: _tech_only(d)) if full_validation else None,
                          causality_data=tgt[["open", "high", "low", "close", "adj_close", "volume"]] if full_validation else None)

        # --------------------------------------------------- training-window choice
        scheme = self.s.models.training_window
        window_cmp = {}
        if scheme == "auto":
            scheme, window_cmp = self.compare_training_windows(X_all, labels[primary], primary)

        results: dict[int, HorizonResult] = {}
        n_h = len(horizons)
        for hi, h in enumerate(horizons):
            self.progress(f"Forecasting horizon {h}D ({hi + 1}/{n_h})", hi / n_h)
            try:
                results[h] = self._run_horizon(h, X_all, tgt, labels[h], regimes, models, scheme, tracker, as_of, fs)
            except Exception as exc:  # noqa: BLE001
                log.exception("horizon %s failed", h)
                hr = HorizonResult(h, str(as_of.date()))
                hr.warnings.append(f"Horizon {h}D failed: {exc}")
                results[h] = hr
            for n, why in skipped.items():
                results[h].model_errors[n] = f"skipped: {why}"
            if audit.passed is False:
                results[h].warnings.append("Leakage audit FAILED - forecasts are not trustworthy.")
        sel = tracker.table()
        res = ForecastResult(str(as_of.date()), results, audit, sel, scheme, window_cmp,
                             {"total_s": round(time.perf_counter() - t_start, 1)})
        if register and self.registry:
            self._register(res, fs, X_all)
        self.progress("Forecasting complete", 1.0)
        return res

    # ------------------------------------------------------------------ per horizon
    def _run_horizon(self, h, X_all, tgt, lab, regimes, models, scheme, tracker, as_of, fs) -> HorizonResult:
        hr = HorizonResult(h, str(as_of.date()), interval_level=1 - self.s.models.conformal_alpha)
        warm = 252
        y_ret, y_up, le = lab["fwd_ret"], lab["direction"], lab["label_end"]
        # only rows whose label window has fully ended by the as-of date are trainable
        trainable = le.notna() & (le <= as_of)
        X = X_all.iloc[warm:]
        y_ret, y_up, le_t, dd = y_ret.reindex(X.index), y_up.reindex(X.index), le.reindex(X.index), lab["dd10"].reindex(X.index)
        mask_tr = trainable.reindex(X.index).fillna(False).values
        le_masked = le_t.where(mask_tr)
        n_lab = int(mask_tr.sum())
        hr.n_train = n_lab
        min_train = max(500, min(1500, int(n_lab * 0.4)))
        if n_lab < 600:
            hr.warnings.append(f"Insufficient labelled history for {h}D ({n_lab} rows < 600); horizon skipped.")
            return hr
        n_splits = self.s.models.n_splits if n_lab > 2000 else max(2, self.s.models.n_splits - 2)
        embargo = max(self.s.models.embargo_days, min(h, 20))
        splits = list(walk_forward(X.index, le_masked, n_splits, min_train, embargo=embargo))
        if not splits:
            hr.warnings.append(f"No valid walk-forward folds for {h}D.")
            return hr
        weights_all = sample_weights(X.index, scheme, self.s.models.exp_halflife_years, self.s.models.recent_window_years)
        if h > 60:  # specialised model sets per horizon
            dl = [m for m in models if ALL_MODELS[m].info.family == "deep_learning"]
            for m in dl:
                hr.model_errors[m] = (f"not used for {h}D: overlapping {h}-day labels leave too few independent samples "
                                      "for sequence models (benchmarked on horizons <= 60D)")
            models = [m for m in models if m not in dl]
        oos_ret = pd.DataFrame(np.nan, index=X.index, columns=models)
        oos_up = pd.DataFrame(np.nan, index=X.index, columns=models)
        oos_dd = pd.Series(np.nan, index=X.index)
        fold_info = []
        context_close = tgt["close"]
        for sp in splits:
            Xtr, Xte = X.iloc[sp.train], X.iloc[sp.test]
            sel, _ = select_features(Xtr, y_ret.iloc[sp.train], self.s.models.max_features, self.s.models.random_seed,
                                     use_stability=False)
            reg_name = regimes["regime_rule"].reindex(Xtr.index).dropna().iloc[-1] if regimes is not None and \
                "regime_rule" in regimes else None
            tracker.update(sel, reg_name)
            sel = sel + [c for c in REGIME_FEATURES if c in X.columns and c not in sel]
            w = weights_all[sp.train] / weights_all[sp.train].mean()
            fold_hp = self._tune(models, Xtr[sel], y_ret.iloc[sp.train], y_up.iloc[sp.train], le_masked.iloc[sp.train], h, hr)
            for name in models:
                try:
                    m = self._make(name, h, fold_hp.get(name))
                    if m.needs_context:
                        m.set_context(X[sel] if m.info.family == "deep_learning" else None, context_close)
                    m.fit(Xtr[sel], y_ret.iloc[sp.train], y_up.iloc[sp.train], w)
                    p = m.predict(Xte[sel])
                    oos_ret.loc[Xte.index, name] = p["ret"].values
                    oos_up.loc[Xte.index, name] = p["p_up"].values
                except Exception as exc:  # noqa: BLE001
                    hr.model_errors[name] = f"fold {sp.fold}: {type(exc).__name__}: {exc}"[:300]
                    log.debug(traceback.format_exc())
            try:
                oos_dd.loc[Xte.index] = _dd_classifier(Xtr[sel], dd.iloc[sp.train], Xte[sel], self.s.models.random_seed)
            except Exception:
                pass
            fold_info.append(sp.describe(X.index))
        hr.folds = fold_info
        good = [m for m in models if oos_ret[m].notna().sum() > 0.5 * sum(len(s.test) for s in splits)]
        if not good:
            hr.warnings.append("No model produced out-of-sample predictions.")
            return hr
        oos_rows = oos_ret[good].dropna(how="all").index
        yr, yu = y_ret.reindex(oos_rows), y_up.reindex(oos_rows)
        regime_lab = regimes["regime_rule"].reindex(oos_rows) if regimes is not None and "regime_rule" in regimes else None
        naive_err = (oos_ret.loc[oos_rows, "naive_drift"] - yr) if "naive_drift" in good else None
        for name in good:
            pr, pu = oos_ret.loc[oos_rows, name], oos_up.loc[oos_rows, name]
            mt = {**regression_metrics(yr, pr), **{k: v for k, v in classification_metrics(yu, pu).items()
                                                   if k in ("accuracy", "precision", "recall", "f1", "auc", "brier",
                                                            "log_loss", "ece")}}
            mt.update(_strategy_stats(pr, yr, h))
            hr.model_scores[name] = mt
            if naive_err is not None and name != "naive_drift":
                hr.dm_tests[name] = diebold_mariano(pr - yr, naive_err, h)
            if regime_lab is not None:
                hr.scores_by_regime[name] = {
                    rg: {"directional_accuracy": float((np.sign(pr[regime_lab == rg]) == np.sign(yr[regime_lab == rg])).mean()),
                         "brier": brier(yu[regime_lab == rg], pu[regime_lab == rg]), "n": int((regime_lab == rg).sum())}
                    for rg in regime_lab.dropna().unique() if (regime_lab == rg).sum() >= 30}

        # ----------------------------------------------- ensemble comparison (OOS, non-overlapping)
        state = regimes[[c for c in REGIME_FEATURES if c in regimes]].reindex(oos_rows) if regimes is not None else None
        cmp_ = walk_forward_ensembles(oos_ret.loc[oos_rows, good], oos_up.loc[oos_rows, good], yr, yu,
                                      le_t.reindex(oos_rows), state, step=max(21, h), min_hist=250)
        hr.ensemble_comparison = {m: {k: v for k, v in d.items() if not k.startswith("pred_")} for m, d in cmp_.items()}
        scored = {m: _composite(d) for m, d in cmp_.items() if d.get("n", 0) > 50}
        if scored:
            # choose on the first 2/3 of the ensemble-evaluation period, report the rest as honest hold-out
            hr.ensemble_method = _choose_method_honestly(cmp_, yr, yu)
        ens_r = cmp_[hr.ensemble_method]["pred_ret"] if scored else oos_ret.loc[oos_rows, good].mean(axis=1)
        ens_u = cmp_[hr.ensemble_method]["pred_up"] if scored else oos_up.loc[oos_rows, good].mean(axis=1)
        ok = ens_r.notna() & yr.notna()

        # ----------------------------------------------- calibration (fit early OOS, evaluate late OOS)
        cut = int(ok.sum() * 0.6)
        idx_ok = ens_r[ok].index
        cal = ProbabilityCalibrator().fit(ens_u[idx_ok[:cut]].values, yu[idx_ok[:cut]].values)
        late = idx_ok[cut:]
        p_late_cal = cal.transform(ens_u[late].values)
        hr.calibration = {"method": cal.chosen_, **calibration_report(yu[late].values, ens_u[late].values, p_late_cal),
                          "n_eval": int(len(late))}
        from ..validation.metrics import reliability

        hr.calibration_curve = reliability(yu[late].values, p_late_cal).to_dict("records")
        final_cal = ProbabilityCalibrator().fit(ens_u[idx_ok].values, yu[idx_ok].values)
        dd_ok = oos_dd.reindex(idx_ok).notna() & dd.reindex(idx_ok).notna()
        dd_cal = ProbabilityCalibrator("isotonic").fit(oos_dd.reindex(idx_ok)[dd_ok].values, dd.reindex(idx_ok)[dd_ok].values) \
            if dd_ok.sum() > 200 else None

        # ----------------------------------------------- conformal (OOS residuals, coverage checked honestly)
        # residual window grows with the horizon: overlapping h-day residuals carry ~window/h independent
        # observations, so long horizons need (much) longer windows to avoid one regime biasing the interval
        cwin = max(750, 12 * h)
        cbt = rolling_conformal_backtest(yr[idx_ok], ens_r[idx_ok], self.s.models.conformal_alpha, window=cwin,
                                         min_n=150, gap=h + 1)
        hr.conformal = interval_metrics(cbt["y"], cbt["lo"], cbt["hi"], 1 - self.s.models.conformal_alpha)
        conf = ConformalRegressor(self.s.models.conformal_alpha, window=cwin).fit(yr[idx_ok].values, ens_r[idx_ok].values)

        hr.oos = pd.DataFrame({"y_ret": yr, "y_up": yu, "ens_ret": ens_r, "ens_up": ens_u,
                               "ens_up_cal": pd.Series(final_cal.transform(ens_u.fillna(0.5).values), index=ens_u.index)
                               .where(ens_u.notna()), "ci_lo": cbt["lo"].reindex(oos_rows), "ci_hi": cbt["hi"].reindex(oos_rows),
                               "regime": regime_lab}).join(oos_ret.loc[oos_rows, good].add_prefix("m_"))
        hr.n_oos = int(ok.sum())

        # ----------------------------------------------- final refit on all labelled history
        Xtrain = X[mask_tr]
        sel, scores = select_features(Xtrain, y_ret[mask_tr], self.s.models.max_features, self.s.models.random_seed,
                                      use_stability=True)
        hr.feature_scores = scores
        hr.selected_features = sel
        sel = sel + [c for c in REGIME_FEATURES if c in X.columns and c not in sel]
        w = weights_all[mask_tr] / weights_all[mask_tr].mean()
        x_now = X_all.loc[[as_of]] if as_of in X_all.index else X_all.iloc[[-1]]
        final_hp = self._tune(good, Xtrain[sel], y_ret[mask_tr], y_up[mask_tr], le_masked[mask_tr], h, hr)
        preds_now = {}
        for name in good:
            try:
                m = self._make(name, h, final_hp.get(name))
                if m.needs_context:
                    m.set_context(X_all[sel] if m.info.family == "deep_learning" else None, tgt["close"])
                m.fit(Xtrain[sel], y_ret[mask_tr], y_up[mask_tr], w)
                p = m.predict(x_now[sel])
                preds_now[name] = {"ret": float(p["ret"].iloc[0]), "p_up": float(p["p_up"].iloc[0]),
                                   "resid_std": float(m.resid_std_) if np.isfinite(m.resid_std_) else None}
                hr.final_models[name] = m
            except Exception as exc:  # noqa: BLE001
                hr.model_errors[name] = f"final fit: {type(exc).__name__}: {exc}"[:300]
        if not preds_now:
            hr.warnings.append("All final model fits failed.")
            return hr
        hr.model_preds = preds_now
        names_now = list(preds_now)
        Pr = pd.DataFrame([[preds_now[n]["ret"] for n in names_now]], columns=names_now, index=x_now.index)
        Pu = pd.DataFrame([[preds_now[n]["p_up"] for n in names_now]], columns=names_now, index=x_now.index)
        hist_err = oos_ret.loc[idx_ok, names_now].sub(yr[idx_ok], axis=0)
        # hist rows must have labels ended before as-of (true by construction of idx_ok)
        S_hist = state.reindex(idx_ok).fillna(0) if state is not None else None
        S_now = regimes[[c for c in REGIME_FEATURES if c in regimes]].reindex(x_now.index).fillna(0) if regimes is not None else None
        try:
            r_now, u_now, info = combine(hr.ensemble_method, Pr, Pu, hist_err, yr[idx_ok], oos_ret.loc[idx_ok, names_now].fillna(0),
                                         oos_up.loc[idx_ok, names_now].fillna(0.5), yu[idx_ok], S_hist, S_now)
        except Exception as exc:  # noqa: BLE001
            hr.warnings.append(f"Ensemble '{hr.ensemble_method}' failed ({exc}); equal weight used.")
            hr.ensemble_method = "equal_weight"
            r_now, u_now, info = combine("equal_weight", Pr, Pu, None)
        hr.ensemble_weights = {k: round(float(v), 4) for k, v in info.get("weights", {}).items()}
        hr.expected_return = float(np.ravel(r_now)[0])
        hr.p_up_raw = float(np.clip(np.ravel(u_now)[0], 0.01, 0.99))
        hr.p_up = float(final_cal.transform([hr.p_up_raw])[0])
        hr.p_down = 1 - hr.p_up
        hr.interval = tuple(float(v) for v in conf.interval(hr.expected_return))
        med_res = float(np.median(conf.resid_)) if conf.n else 0.0
        if np.isfinite(hr.interval[0]) and not (hr.interval[0] <= hr.expected_return <= hr.interval[1]):
            hr.warnings.append(
                f"Residual bias: the ensemble has systematically {'under' if med_res > 0 else 'over'}-predicted "
                f"{h}D returns out-of-sample (median residual {med_res:+.1%}, n_eff~{conf.n // max(h, 1)}); the point "
                "forecast lies outside its own conformal interval - treat the point forecast as unreliable.")
        if conf.n and conf.n // max(h, 1) < 20:
            hr.warnings.append(f"Only ~{conf.n // max(h, 1)} independent {h}D outcomes support the interval "
                               "(overlapping labels): interval and probabilities are imprecise.")
        hr.conformal["median_residual"] = med_res
        hr.conformal["effective_independent_residuals"] = int(conf.n // max(h, 1))
        hr.quantiles = conf.quantiles(hr.expected_return)
        hr.p_gt5 = conf.prob_above(hr.expected_return, 0.05)
        hr.p_lt5 = conf.prob_below(hr.expected_return, -0.05)
        try:
            raw_dd = _dd_classifier(Xtrain[sel], dd[mask_tr], x_now[sel], self.s.models.random_seed)[0]
            hr.p_dd10 = float(dd_cal.transform([raw_dd])[0]) if dd_cal is not None else float(raw_dd)
        except Exception:
            hr.p_dd10 = np.nan

        # ----------------------------------------------- uncertainty / disagreement
        rets = np.array([preds_now[n]["ret"] for n in names_now if n != "naive_drift"])
        if len(rets) >= 2:
            sign_agree = max((rets > 0).mean(), (rets < 0).mean())
            spread = float(np.std(rets))
        else:
            sign_agree, spread = np.nan, np.nan
        aleatoric = float(np.std(conf.resid_)) if conf.n else np.nan
        hr.disagreement = {"std_of_model_returns": spread, "sign_agreement": float(sign_agree) if not np.isnan(sign_agree) else None,
                           "range": [float(rets.min()), float(rets.max())] if len(rets) else None, "n_models": int(len(rets))}
        epi_ratio = spread / aleatoric if aleatoric and np.isfinite(spread) else np.nan
        hr.uncertainty = {"aleatoric_std": aleatoric, "epistemic_std": spread,
                          "epistemic_share": float(spread ** 2 / (spread ** 2 + aleatoric ** 2)) if aleatoric and np.isfinite(spread) else None}
        # confidence: OOS skill x calibration quality x model agreement (bounded, never 100 %)
        ens_sc = hr.ensemble_comparison.get(hr.ensemble_method, {})
        naive_da = hr.model_scores.get("naive_drift", {}).get("directional_accuracy") or 0.5
        # skill must be measured against the naive drift benchmark (base rate), not against 50 %
        da_skill = np.clip(((ens_sc.get("directional_accuracy") or 0.5) - max(0.5, naive_da)) / 0.08, 0, 1)
        ic_skill = np.clip((ens_sc.get("ic_spearman") or 0.0) / 0.15, 0, 1)
        skill = 0.5 * da_skill + 0.5 * ic_skill
        calq = np.clip(1 - (hr.calibration.get("calibrated", {}).get("ece") or 0.1) / 0.15, 0, 1)
        agree = sign_agree if not np.isnan(sign_agree) else 0.5
        disp_pen = np.clip(1 - (epi_ratio if np.isfinite(epi_ratio) else 0.5), 0.2, 1)
        hr.confidence = float(np.clip(0.15 + 0.35 * skill + 0.2 * calq + 0.2 * (agree - 0.5) * 2 * disp_pen, 0.05, 0.9))
        if any(w.startswith("Residual bias") for w in hr.warnings):
            hr.confidence *= 0.5
        if hr.n_oos < 300:
            hr.warnings.append(f"Only {hr.n_oos} OOS observations - statistics have wide error bars.")
        return hr

    def _tune(self, models, X, y_ret, y_up, le, h, hr) -> dict:
        """Nested HPO on the given (training-only) window. Returns {model: params}."""
        out = {}
        # nested HPO is run for the primary horizon only (cost); other horizons use regularised defaults
        if not self.hpo_trials or h != self.s.models.primary_horizon:
            return out
        from .hpo import tune

        for name in ("lightgbm", "xgboost"):
            if name in models:
                try:
                    r = tune(name, X, y_ret, y_up, le, h, self.hpo_trials, self.s.models.random_seed)
                    out[name] = r.get("params", {})
                    hr.hpo.setdefault(name, []).append(r)
                except Exception as exc:  # noqa: BLE001
                    hr.warnings.append(f"HPO for {name} failed: {exc}")
        return out

    # ------------------------------------------------------------------ training window
    def compare_training_windows(self, X_all: pd.DataFrame, lab: pd.DataFrame, h: int) -> tuple[str, dict]:
        """Walk-forward comparison of full / recent / exponentially-weighted training."""
        from .tree.models import HistGBForecaster

        X = X_all.iloc[252:]
        y_ret, y_up, le = lab["fwd_ret"].reindex(X.index), lab["direction"].reindex(X.index), lab["label_end"].reindex(X.index)
        le = le.where(le <= X.index[-1])
        out = {}
        splits = list(walk_forward(X.index, le, 4, max(500, int(le.notna().sum() * 0.4)), embargo=max(5, h)))
        for scheme in ("full", "recent", "exp_weighted"):
            w_all = sample_weights(X.index, scheme, self.s.models.exp_halflife_years, self.s.models.recent_window_years)
            preds, ys, us, ps = [], [], [], []
            for sp in splits:
                tr = sp.train[w_all[sp.train] > 0]
                if len(tr) < 300:
                    continue
                m = HistGBForecaster(h, self.s.models.random_seed).fit(X.iloc[tr], y_ret.iloc[tr], y_up.iloc[tr], w_all[tr])
                p = m.predict(X.iloc[sp.test])
                preds.append(p["ret"])
                ps.append(p["p_up"])
                ys.append(y_ret.iloc[sp.test])
                us.append(y_up.iloc[sp.test])
            if preds:
                mt = regression_metrics(pd.concat(ys), pd.concat(preds))
                mt["brier"] = brier(pd.concat(us), pd.concat(ps))
                out[scheme] = mt
        if not out:
            return "full", out
        best = max(out, key=lambda k: _composite(out[k]))
        return best, out

    # ------------------------------------------------------------------ registry
    def _register(self, res: ForecastResult, fs: FeatureSet, X_all: pd.DataFrame) -> None:
        for h, hr in res.horizons.items():
            for name, m in hr.final_models.items():
                val = hr.model_scores.get(name, {})
                try:
                    hr.model_ids[name] = self.registry.register(
                        name, h, X_all.index[0], res.as_of, fs.version, fs.dataset_version,
                        {**getattr(m, "hp", {}), "training_window": res.training_window},
                        {k: v for k, v in val.items() if isinstance(v, (int, float))},
                        m if ALL_MODELS[name].info.family != "deep_learning" else None)
                except Exception as exc:  # noqa: BLE001
                    log.warning("registry failed for %s: %s", name, exc)
            if hr.model_preds:
                hr.model_ids["ensemble"] = self.registry.register(
                    "ensemble", h, X_all.index[0], res.as_of, fs.version, fs.dataset_version,
                    {"method": hr.ensemble_method, "weights": hr.ensemble_weights},
                    {k: v for k, v in hr.ensemble_comparison.get(hr.ensemble_method, {}).items() if isinstance(v, (int, float))},
                    None, role="production")


def _strategy_stats(pred: pd.Series, y: pd.Series, h: int) -> dict:
    """Long/flat on non-overlapping h-period blocks (a quick economic-value check, before costs)."""
    d = pd.DataFrame({"p": pred, "y": y}).dropna().iloc[::max(1, h)]
    if len(d) < 10:
        return {}
    strat = np.where(d["p"] > 0, d["y"], 0.0)
    per_year = 252 / h
    mu, sd = strat.mean(), strat.std()
    return {"strat_sharpe": float(mu / sd * np.sqrt(per_year)) if sd > 0 else np.nan,
            "buyhold_sharpe": float(d["y"].mean() / d["y"].std() * np.sqrt(per_year)) if d["y"].std() > 0 else np.nan,
            "exposure": float((d["p"] > 0).mean())}


def _choose_method_honestly(cmp_: dict, yr: pd.Series, yu: pd.Series) -> str:
    """Pick the ensemble method on the first two-thirds of its OOS period only."""
    best, best_s = "equal_weight", -np.inf
    for m in METHODS:
        r, p = cmp_[m]["pred_ret"], cmp_[m]["pred_up"]
        ok = r.notna()
        idx = r[ok].index
        if len(idx) < 90:
            continue
        sel = idx[: int(len(idx) * 2 / 3)]
        sc = _composite({**regression_metrics(yr[sel], r[sel]), "brier": brier(yu[sel], p[sel])})
        if sc > best_s:
            best, best_s = m, sc
    return best


def _dd_classifier(Xtr: pd.DataFrame, y: pd.Series, Xte: pd.DataFrame, seed: int) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from .base import Preprocessor

    m = y.notna().values
    yy = y.values[m].astype(int)
    if len(np.unique(yy)) < 2:
        return np.full(len(Xte), float(yy.mean()) if len(yy) else np.nan)
    pre = Preprocessor().fit(Xtr)
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05, min_samples_leaf=80,
                                         random_state=seed).fit(pre.transform(Xtr)[m], yy)
    return clf.predict_proba(pre.transform(Xte))[:, 1]


def _tech_only(d: pd.DataFrame) -> pd.DataFrame:
    from ..features.technical.indicators import technical_features

    return technical_features(d)
