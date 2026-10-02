"""Detectors, stacking ensemble and baseline classifiers.

Each detector sees one feature family (or all features) and produces
    score       calibrated probability (Platt scaling on out-of-fold scores)
    confidence  agreement of its cross-validation committee and feature coverage
    features    per-feature contributions (exact for the linear detectors)
    error       committee standard deviation + its validation Brier score

A meta-classifier (logistic regression on detector logits) learns the ensemble
weights from out-of-fold predictions; weights are never hard-coded. Detectors
that receive negative weight (redundant/collinear) are pruned and the stacker
is refitted so the reported weights are interpretable.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .calibration import PlattCalibrator, balanced_weights, brier, fit_best_calibrator, logit
from .features import DETECTOR_FAMILIES, family_of

DETECTOR_SPECS: dict[str, tuple[str, list[str] | None]] = {
    **{fam: ("logreg", [fam]) for fam in DETECTOR_FAMILIES},
    "statistical": ("hgb", None),
    "neural": ("mlp", None),
}
DETECTOR_LABELS = {
    "perplexity": "Perplexity model", "burstiness": "Burstiness model", "stylometric": "Stylometric model",
    "vocabulary": "Vocabulary model", "syntax": "Syntax model", "semantic": "Semantic model",
    "repetition": "Repetition & transitions model", "statistical": "Statistical model (gradient boosting, all features)",
    "neural": "Neural network (MLP, all features)", "transformer": "Transformer classifier",
}


def _imputer():
    return SimpleImputer(strategy="median", keep_empty_features=True)


def make_estimator(kind: str, seed: int = 0):
    if kind == "logreg":
        return Pipeline([("imp", _imputer()), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))])
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                                              l2_regularization=1.0, class_weight="balanced", random_state=seed)
    if kind == "mlp":
        return Pipeline([("imp", _imputer()), ("sc", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                               max_iter=400, random_state=seed))])
    if kind == "rf":
        return Pipeline([("imp", _imputer()),
                         ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                        class_weight="balanced_subsample", random_state=seed))])
    if kind == "svm":
        return Pipeline([("imp", _imputer()), ("sc", StandardScaler()),
                         ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed))])
    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, class_weight="balanced", random_state=seed, verbose=-1)
    raise ValueError(kind)


def _balanced_resample(X, y, seed):
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    n = max(len(idx0), len(idx1))
    idx = np.concatenate([rng.choice(idx0, n, replace=len(idx0) < n), rng.choice(idx1, n, replace=len(idx1) < n)])
    return X[idx], y[idx]


def fit_estimator(est, kind, X, y, seed=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "mlp":
            Xb, yb = _balanced_resample(X, y, seed)
            est.fit(Xb, yb)
        else:
            est.fit(X, y)
    return est


def predict_scores(est, X) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    d = est.decision_function(X)
    return 1 / (1 + np.exp(-d))


class Detector:
    def __init__(self, name: str, kind: str, feature_names: list[str], all_names: list[str], seed: int = 0):
        self.name = name
        self.kind = kind
        self.feature_names = feature_names
        self.cols = np.array([all_names.index(f) for f in feature_names], dtype=int)
        self.seed = seed
        self.models: list = []
        self.platt: PlattCalibrator | None = None
        self.validation: dict = {}
        self.feature_lo = self.feature_hi = None

    def fit(self, X: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        Xs = X[:, self.cols]
        oof = np.full(len(y), np.nan)
        self.models = []
        for k, (tr, te) in enumerate(folds):
            est = fit_estimator(make_estimator(self.kind, self.seed + k), self.kind, Xs[tr], y[tr], self.seed + k)
            oof[te] = predict_scores(est, Xs[te])
            self.models.append(est)
        self.platt = PlattCalibrator().fit(oof, y, balanced_weights(y))
        cal = self.platt.transform(oof)
        w = balanced_weights(y)
        from sklearn.metrics import roc_auc_score

        self.validation = {"oof_roc_auc": float(roc_auc_score(y, oof)), "oof_brier_balanced": brier(y, cal, w)}
        self.feature_lo = np.nanpercentile(Xs, 0.5, axis=0)
        self.feature_hi = np.nanpercentile(Xs, 99.5, axis=0)
        return cal

    def raw_committee(self, X: np.ndarray) -> np.ndarray:
        Xs = X[:, self.cols]
        return np.vstack([predict_scores(m, Xs) for m in self.models])  # (k, n)

    def predict(self, X: np.ndarray) -> dict:
        C = self.raw_committee(X)
        raw = C.mean(axis=0)
        cal = self.platt.transform(raw)
        cal_members = np.vstack([self.platt.transform(c) for c in C])
        std = cal_members.std(axis=0)
        Xs = X[:, self.cols]
        with np.errstate(invalid="ignore"):
            inside = ((Xs >= self.feature_lo) & (Xs <= self.feature_hi)) | np.isnan(Xs)
        coverage = inside.mean(axis=1)
        conf = np.clip(1 - std / 0.25, 0, 1) * np.clip(coverage, 0, 1) ** 2
        return {"prob": cal, "std": std, "confidence": conf, "coverage": coverage}

    def contributions(self, X: np.ndarray) -> np.ndarray | None:
        """Per-feature logit contributions coef * standardized value (linear detectors only)."""
        if self.kind != "logreg":
            return None
        Xs = X[:, self.cols]
        acc = np.zeros(Xs.shape, dtype=float)
        for m in self.models:
            Z = m[:-1].transform(Xs)
            acc += Z * m[-1].coef_[0]
        slope = self.platt.params["slope"] if self.platt else 1.0
        return slope * acc / len(self.models)

    def standardized(self, X: np.ndarray) -> np.ndarray | None:
        if self.kind != "logreg":
            return None
        return self.models[0][:-1].transform(X[:, self.cols])


class Ensemble:
    """Stacked ensemble for one level ("document" or "window")."""

    def __init__(self, level: str, feature_names: list[str], seed: int = 0, folds: int = 5,
                 detectors: dict[str, tuple[str, list[str] | None]] | None = None):
        self.level = level
        self.feature_names = list(feature_names)
        self.seed = seed
        self.n_folds = folds
        specs = detectors or DETECTOR_SPECS
        self.detectors: dict[str, Detector] = {}
        for name, (kind, fams) in specs.items():
            names = self.feature_names if fams is None else [f for f in self.feature_names if family_of(f) in fams]
            if names:
                self.detectors[name] = Detector(name, kind, names, self.feature_names, seed)
        self.stacker: LogisticRegression | None = None
        self.stack_names: list[str] = []
        self.text_detectors: dict = {}  # name -> object with predict_texts(list[str]) -> np.ndarray
        self.calibrator = None
        self.calibration_report: dict = {}
        self.population: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, verbose: bool = True) -> "Ensemble":
        y = np.asarray(y, int)
        cv = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        folds = list(cv.split(X, y, groups))
        oof = {}
        for name, det in self.detectors.items():
            oof[name] = det.fit(X, y, folds)
            if verbose:
                print(f"    [{self.level}] {name:12s} OOF AUC={det.validation['oof_roc_auc']:.3f}", flush=True)
        self._fit_stacker(oof, y)
        self.population = population_stats(X, y, self.feature_names)
        return self

    def _fit_stacker(self, oof: dict[str, np.ndarray], y: np.ndarray) -> None:
        names = list(oof)
        while True:
            Z = np.column_stack([logit(oof[n]) for n in names])
            st = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Z, y)
            coef = st.coef_[0]
            if (coef > 0).all() or len(names) == 1:
                break
            names = [n for n, c in zip(names, coef) if c > 0] or [names[int(np.argmax(coef))]]
        self.stacker = st
        self.stack_names = names

    def detector_outputs(self, X: np.ndarray, texts: list[str] | None = None) -> dict[str, dict]:
        outs = {name: det.predict(X) for name, det in self.detectors.items()}
        for name, td in getattr(self, "text_detectors", {}).items():
            if texts is None:
                raise ValueError(f"detector '{name}' needs the raw texts")
            p = np.asarray(td.predict_texts(texts), dtype=float)
            outs[name] = {"prob": p, "std": np.zeros_like(p), "confidence": np.ones_like(p), "coverage": np.ones_like(p)}
        return outs

    def stack_raw(self, outs: dict[str, dict]) -> np.ndarray:
        Z = np.column_stack([logit(outs[n]["prob"]) for n in self.stack_names])
        return self.stacker.predict_proba(Z)[:, 1]

    def predict_raw(self, X: np.ndarray, texts: list[str] | None = None) -> np.ndarray:
        return self.stack_raw(self.detector_outputs(X, texts))

    def calibrate(self, X: np.ndarray, y: np.ndarray, methods=("platt", "isotonic"), texts: list[str] | None = None) -> dict:
        raw = self.predict_raw(X, texts)
        self.calibrator, self.calibration_report = fit_best_calibrator(raw, np.asarray(y, int), methods, seed=self.seed)
        return self.calibration_report

    def add_text_detector(self, name: str, detector, X_a: np.ndarray, texts_a: list[str], y_a: np.ndarray,
                          X_b: np.ndarray, texts_b: list[str], y_b: np.ndarray) -> dict:
        """Add a text-based detector (e.g. a fine-tuned transformer).

        Its training data never overlaps the calibration split, but the stacker was fitted on
        out-of-fold feature-detector scores; to avoid mixing regimes the stacker is refitted on
        calibration half A (all detectors out-of-sample there) and the final calibrator on half B.
        """
        if not hasattr(self, "text_detectors"):
            self.text_detectors = {}
        self.text_detectors[name] = detector
        outs = self.detector_outputs(X_a, texts_a)
        self._fit_stacker({n: outs[n]["prob"] for n in outs}, np.asarray(y_a, int))
        return self.calibrate(X_b, y_b, texts=texts_b)

    def predict_proba(self, X: np.ndarray, texts: list[str] | None = None) -> np.ndarray:
        return self.calibrator.transform(self.predict_raw(X, texts))

    def predict_detail(self, X: np.ndarray, texts: list[str] | None = None) -> dict:
        outs = self.detector_outputs(X, texts)
        raw = self.stack_raw(outs)
        return {"detectors": outs, "raw": raw, "prob": self.calibrator.transform(raw)}

    @property
    def weights(self) -> dict[str, float]:
        coef = self.stacker.coef_[0]
        tot = coef.sum()
        w = {n: float(c / tot) for n, c in zip(self.stack_names, coef)}
        return {n: w.get(n, 0.0) for n in list(self.detectors) + list(getattr(self, "text_detectors", {}))}


def population_stats(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """Per-feature reference distributions (human / AI) for explanations and OOD checks."""
    qs = np.linspace(0, 100, 21)
    out = {}
    for j, n in enumerate(names):
        col = X[:, j]
        entry = {}
        for lab, key in ((0, "human"), (1, "ai")):
            v = col[(y == lab) & ~np.isnan(col)]
            if len(v) < 5:
                continue
            entry[key] = {"mean": float(v.mean()), "std": float(v.std() + 1e-12),
                          "quantiles": np.percentile(v, qs).tolist()}
        allv = col[~np.isnan(col)]
        if len(allv):
            entry["all"] = {"q005": float(np.percentile(allv, 0.5)), "q995": float(np.percentile(allv, 99.5))}
        out[n] = entry
    return out


def percentile_in(ref_quantiles: list[float], value: float) -> float:
    """Approximate percentile (0-100) of ``value`` within a reference distribution."""
    q = np.asarray(ref_quantiles)
    if value != value:
        return float("nan")
    return float(np.interp(value, q, np.linspace(0, 100, len(q)), left=0.0, right=100.0))


BASELINES = ["logreg", "rf", "hgb", "lgbm", "svm", "mlp"]
BASELINE_LABELS = {"logreg": "Logistic Regression", "rf": "Random Forest", "hgb": "HistGradientBoosting",
                   "lgbm": "LightGBM", "svm": "SVM (RBF)", "mlp": "Neural network (MLP)"}


def compare_baselines(Xtr, ytr, Xte, yte, Xcal=None, ycal=None, seed: int = 0, svm_max: int = 8000) -> dict:
    """Train each baseline on the training split, Platt-calibrate on the calibration split, test once."""
    from .metrics import binary_metrics

    res = {}
    for kind in BASELINES:
        try:
            est = make_estimator(kind, seed)
        except ImportError:
            res[kind] = {"skipped": "dependency not installed"}
            continue
        X, y = Xtr, ytr
        if kind == "svm" and len(y) > svm_max:
            idx = np.random.default_rng(seed).choice(len(y), svm_max, replace=False)
            X, y = X[idx], y[idx]
        fit_estimator(est, kind, X, y, seed)
        p = predict_scores(est, Xte)
        if Xcal is not None:
            cal = PlattCalibrator().fit(predict_scores(est, Xcal), ycal, balanced_weights(ycal))
            p = cal.transform(p)
        res[kind] = binary_metrics(yte, p)
    return res
