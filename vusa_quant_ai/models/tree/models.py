"""Machine-learning forecasters: Elastic Net, Random Forest, Extra Trees,
HistGradientBoosting, XGBoost, LightGBM, CatBoost.

Each model fits a regressor (expected forward return) and a classifier (P(return>0)).
Optional libraries are detected at runtime; if unavailable, the model reports itself as
unavailable instead of silently substituting another algorithm.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from ..base import BaseForecaster, ModelInfo, Preprocessor


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


class SklearnPairForecaster(BaseForecaster):
    default_hp: dict = {}

    def _make(self, hp):  # -> (regressor, classifier)
        raise NotImplementedError

    def fit(self, X, y_ret, y_up, w=None):
        hp = {**self.default_hp, **self.hp}
        self.pre_ = Preprocessor().fit(X)
        Z = self.pre_.transform(X)
        m = y_ret.notna().values & y_up.notna().values
        Z, yr, yc = Z[m], y_ret.values[m], y_up.values[m].astype(int)
        ww = None if w is None else np.asarray(w, float)[m]
        if ww is not None:
            keep = ww > 0  # "recent window" scheme: drop zero-weight rows instead of weighting
            Z, yr, yc, ww = Z[keep], yr[keep], yc[keep], ww[keep]
            if np.allclose(ww, ww[0]):
                ww = None  # uniform weights: avoid slow weighted binning in some estimators
        self.reg_, self.clf_ = self._make(hp)
        self._fit_one(self.reg_, Z, yr, ww)
        self.single_class_ = len(np.unique(yc)) < 2
        if self.single_class_:
            self.const_p_ = float(yc.mean())
        else:
            self._fit_one(self.clf_, Z, yc, ww)
        self.resid_std_ = float(np.std(yr - self.reg_.predict(Z)))
        return self

    @staticmethod
    def _fit_one(est, Z, y, w):
        try:
            est.fit(Z, y, sample_weight=w) if w is not None else est.fit(Z, y)
        except TypeError:  # estimator without sample_weight support
            est.fit(Z, y)

    def predict(self, X):
        Z = self.pre_.transform(X)
        ret = self.reg_.predict(Z)
        p = np.full(len(Z), self.const_p_) if self.single_class_ else self.clf_.predict_proba(Z)[:, 1]
        return pd.DataFrame({"ret": ret, "p_up": np.clip(p, 0.001, 0.999)}, index=X.index)

    def feature_importance(self) -> pd.Series | None:
        for est in (self.reg_,):
            if hasattr(est, "feature_importances_"):
                return pd.Series(est.feature_importances_, index=self.pre_.cols_)
            if hasattr(est, "coef_"):
                return pd.Series(np.abs(np.ravel(est.coef_)), index=self.pre_.cols_)
        return None


class ElasticNetForecaster(SklearnPairForecaster):
    info = ModelInfo("elastic_net", "ml", "Elastic-net regression + L1/L2 logistic regression")
    default_hp = {"alpha": 0.001, "l1_ratio": 0.5, "C": 0.05}

    def _make(self, hp):
        from sklearn.linear_model import ElasticNet, LogisticRegression

        return (ElasticNet(alpha=hp["alpha"], l1_ratio=hp["l1_ratio"], max_iter=5000, random_state=self.seed),
                LogisticRegression(C=hp["C"], solver="saga", l1_ratio=hp["l1_ratio"],
                                   max_iter=3000, random_state=self.seed))


class RandomForestForecaster(SklearnPairForecaster):
    info = ModelInfo("random_forest", "ml", "Random forest (shallow, heavily regularised)")
    default_hp = {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 50, "max_features": 0.3}

    def _make(self, hp):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        kw = dict(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"],
                  max_features=hp["max_features"], n_jobs=-1, random_state=self.seed)
        return RandomForestRegressor(**kw), RandomForestClassifier(**kw)


class ExtraTreesForecaster(SklearnPairForecaster):
    info = ModelInfo("extra_trees", "ml", "Extremely randomised trees")
    default_hp = {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 50, "max_features": 0.3}

    def _make(self, hp):
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        kw = dict(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"],
                  max_features=hp["max_features"], n_jobs=-1, random_state=self.seed)
        return ExtraTreesRegressor(**kw), ExtraTreesClassifier(**kw)


class HistGBForecaster(SklearnPairForecaster):
    info = ModelInfo("hist_gb", "ml", "Histogram gradient boosting (scikit-learn)")
    default_hp = {"learning_rate": 0.03, "max_iter": 250, "max_depth": 3, "min_samples_leaf": 80, "l2_regularization": 1.0}

    def _make(self, hp):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        kw = dict(learning_rate=hp["learning_rate"], max_iter=hp["max_iter"], max_depth=hp["max_depth"],
                  min_samples_leaf=hp["min_samples_leaf"], l2_regularization=hp["l2_regularization"],
                  random_state=self.seed)
        return HistGradientBoostingRegressor(**kw), HistGradientBoostingClassifier(**kw)


class XGBForecaster(SklearnPairForecaster):
    info = ModelInfo("xgboost", "ml", "XGBoost gradient boosting", available=_has("xgboost"),
                     unavailable_reason="" if _has("xgboost") else "xgboost not installed")
    default_hp = {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.8,
                  "colsample_bytree": 0.6, "min_child_weight": 20, "reg_lambda": 5.0}

    def _make(self, hp):
        import xgboost as xgb

        kw = dict(n_estimators=hp["n_estimators"], learning_rate=hp["learning_rate"], max_depth=hp["max_depth"],
                  subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
                  min_child_weight=hp["min_child_weight"], reg_lambda=hp["reg_lambda"], random_state=self.seed,
                  n_jobs=-1, tree_method="hist", verbosity=0)
        return xgb.XGBRegressor(**kw), xgb.XGBClassifier(**kw, eval_metric="logloss")


class LGBMForecaster(SklearnPairForecaster):
    info = ModelInfo("lightgbm", "ml", "LightGBM gradient boosting", available=_has("lightgbm"),
                     unavailable_reason="" if _has("lightgbm") else "lightgbm not installed")
    default_hp = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 8, "min_child_samples": 60,
                  "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 5.0}

    def _make(self, hp):
        import lightgbm as lgb

        kw = dict(n_estimators=hp["n_estimators"], learning_rate=hp["learning_rate"], num_leaves=hp["num_leaves"],
                  min_child_samples=hp["min_child_samples"], subsample=hp["subsample"], subsample_freq=1,
                  colsample_bytree=hp["colsample_bytree"], reg_lambda=hp["reg_lambda"], random_state=self.seed,
                  n_jobs=-1, verbose=-1)
        return lgb.LGBMRegressor(**kw), lgb.LGBMClassifier(**kw)


class CatBoostForecaster(SklearnPairForecaster):
    info = ModelInfo("catboost", "ml", "CatBoost gradient boosting", available=_has("catboost"),
                     unavailable_reason="" if _has("catboost") else "catboost not installed")
    default_hp = {"iterations": 300, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 5.0}

    def _make(self, hp):
        from catboost import CatBoostClassifier, CatBoostRegressor

        kw = dict(iterations=hp["iterations"], learning_rate=hp["learning_rate"], depth=hp["depth"],
                  l2_leaf_reg=hp["l2_leaf_reg"], random_seed=self.seed, verbose=False, thread_count=-1,
                  allow_writing_files=False)
        return CatBoostRegressor(**kw), CatBoostClassifier(**kw)


class QuantileForecaster:
    """Quantile regression (LightGBM if available, else scikit-learn HGB quantile loss)."""

    def __init__(self, quantiles=(0.1, 0.5, 0.9), seed: int = 42):
        self.quantiles, self.seed = quantiles, seed

    def fit(self, X: pd.DataFrame, y: pd.Series, w=None):
        self.pre_ = Preprocessor().fit(X)
        Z = self.pre_.transform(X)
        m = y.notna().values
        self.models_ = {}
        for q in self.quantiles:
            if _has("lightgbm"):
                import lightgbm as lgb

                est = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=200, learning_rate=0.03,
                                        num_leaves=8, min_child_samples=80, random_state=self.seed, verbose=-1)
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor

                est = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, max_depth=3,
                                                    learning_rate=0.03, random_state=self.seed)
            est.fit(Z[m], y.values[m], sample_weight=None if w is None else np.asarray(w)[m])
            self.models_[q] = est
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.pre_.transform(X)
        out = pd.DataFrame({f"q{int(q * 100)}": m.predict(Z) for q, m in self.models_.items()}, index=X.index)
        # enforce monotone quantiles (no quantile crossing)
        return pd.DataFrame(np.sort(out.values, axis=1), index=out.index, columns=out.columns)


ML_MODELS = {c.info.name: c for c in (ElasticNetForecaster, RandomForestForecaster, ExtraTreesForecaster,
                                      HistGBForecaster, XGBForecaster, LGBMForecaster, CatBoostForecaster)}
