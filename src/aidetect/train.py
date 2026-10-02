"""Training pipeline.

    python train.py --lang en            (after python -m aidetect.build_dataset)

1. Document-level ensemble: family detectors + gradient boosting + MLP, trained
   with grouped stratified K-fold; a logistic meta-classifier learns the weights
   from out-of-fold predictions.
2. Window-level ensemble (sentence scores): same architecture on centred
   5-sentence windows, target = whether the centre sentence is AI-written.
3. Calibration on the held-out calibration split (Platt vs isotonic compared by
   cross-validation; the better one is kept).
4. Band thresholds and the AI-share calibrator are derived from calibration data.
5. Everything is saved to models/<lang>/bundle.joblib and logged to experiments/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from . import FEATURE_VERSION
from .calibration import balanced_weights
from .config import DATA_DIR, MODELS_DIR, load_config
from .data import dataset_version, load_jsonl
from .features import FeatureExtractor, model_feature_names, to_matrix
from .models import Ensemble
from .tracking import log_run


# --------------------------------------------------------------------------- data access
def load_set(data_dir: Path, name: str) -> tuple[dict[str, dict], list[dict]]:
    p_json, p_feat = data_dir / f"{name}.jsonl", data_dir / f"features_{name}.pkl"
    if not p_json.exists() or not p_feat.exists():
        return {}, []
    samples = {s["id"]: s for s in load_jsonl(p_json)}
    with open(p_feat, "rb") as fh:
        feats = [r for r in pickle.load(fh) if "error" not in r and r["id"] in samples]
    return samples, feats


def doc_arrays(samples: dict, feats: list[dict], names: list[str]):
    X = to_matrix([r["doc"] for r in feats], names)
    meta = [samples[r["id"]] for r in feats]
    y = np.array([m["label"] for m in meta], dtype=int)
    groups = np.array([m["group_id"] for m in meta])
    return X, y, groups, meta


def window_arrays(samples: dict, feats: list[dict], names: list[str]):
    Xs, ys, gs, fr, doc_idx, centers = [], [], [], [], [], []
    for i, r in enumerate(feats):
        w = r.get("windows")
        if not w or not w["centers"]:
            continue
        cols = [w["names"].index(n) if n in w["names"] else -1 for n in names]
        X = np.full((len(w["centers"]), len(names)), np.nan)
        for j, c in enumerate(cols):
            if c >= 0:
                X[:, j] = w["X"][:, c]
        Xs.append(X)
        ys.extend(int(v >= 0.5) for v in w["center_label"])
        fr.extend(w["ai_fraction"])
        gs.extend([samples[r["id"]]["group_id"]] * len(w["centers"]))
        doc_idx.extend([i] * len(w["centers"]))
        centers.extend(w["centers"])
    if not Xs:
        return np.zeros((0, len(names))), np.zeros(0, int), np.array([]), np.zeros(0), np.zeros(0, int), np.zeros(0, int)
    return np.vstack(Xs), np.array(ys), np.array(gs), np.array(fr), np.array(doc_idx), np.array(centers)


def first_names(feats: list[dict]) -> tuple[list[str], list[str]]:
    doc_names = model_feature_names(feats[0]["doc"], "document")
    wn = next(r["windows"]["names"] for r in feats if r.get("windows"))
    win_names = model_feature_names({k: 0 for k in wn}, "window")
    return doc_names, win_names


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    cw = np.cumsum(w) / w.sum()
    return float(np.interp(q, cw, v))


def derive_bands(p: np.ndarray, y: np.ndarray, words: np.ndarray, target: float) -> dict:
    human, ai = y == 0, y == 1
    t_high = weighted_quantile(p[human], words[human], 1 - target) if human.any() else 0.7
    t_low = weighted_quantile(p[ai], words[ai], target) if ai.any() else 0.3
    t_high = float(np.clip(t_high, 0.5, 0.9))
    t_low = float(np.clip(t_low, 0.1, 0.5))
    return {"t_high": t_high, "t_low": t_low, "target_human_flag_rate": target,
            "derivation": "t_high: at most target share of human sentence words (calibration split) above it; "
                          "t_low: at most target share of AI sentence words below it"}


def raw_share(probs: np.ndarray, words: np.ndarray) -> float:
    return float(np.sum(probs * words) / max(np.sum(words), 1))


# --------------------------------------------------------------------------- main training routine
def train(lang: str = "en", data_dir: Path | None = None, out_path: Path | None = None, folds: int | None = None,
          max_windows: int = 60000, include_nonnative: bool = False, tag: str = "", verbose: bool = True,
          seed: int | None = None) -> dict:
    cfg = load_config()
    tc = cfg["training"]
    seed = int(tc["seed"] if seed is None else seed)
    folds = int(folds or tc["cv_folds"])
    data_dir = Path(data_dir or DATA_DIR / "processed" / lang)
    t0 = time.time()

    tr_s, tr_f = load_set(data_dir, "train")
    if not tr_f:
        raise SystemExit(f"No featurised training data in {data_dir}. Run: python -m aidetect.build_dataset --lang {lang}")
    sets = {"train": (tr_s, tr_f)}
    for name in ("hybrids_train", "train_augmented", "calib", "hybrids_calib", "nonnative_train"):
        sets[name] = load_set(data_dir, name)
    doc_names, win_names = first_names(tr_f)

    def merged(names):
        s, f = {}, []
        for n in names:
            s.update(sets[n][0])
            f.extend(sets[n][1])
        return s, f

    train_sets = ["train", "hybrids_train", "train_augmented"] + (["nonnative_train"] if include_nonnative else [])
    win_train_sets = ["train", "hybrids_train"] + (["nonnative_train"] if include_nonnative else [])
    s_tr, f_tr = merged(train_sets)
    s_cal, f_cal = merged(["calib", "hybrids_calib"])

    # ---------------- document level
    Xd, yd, gd, _ = doc_arrays(s_tr, f_tr, doc_names)
    if verbose:
        print(f"Document level: {len(yd)} training rows ({yd.sum()} AI), {len(doc_names)} features", flush=True)
    doc_ens = Ensemble("document", doc_names, seed=seed, folds=folds).fit(Xd, yd, gd, verbose=verbose)
    Xdc, ydc, _, meta_c = doc_arrays(s_cal, f_cal, doc_names)
    doc_cal = doc_ens.calibrate(Xdc, ydc, tuple(cfg["calibration"]["methods"]))
    if verbose:
        print(f"  calibration: selected {doc_cal['selected']} | " + ", ".join(
            f"{k}: brier={v['brier']:.4f} ece={v['ece']:.4f}" for k, v in doc_cal["results"].items()), flush=True)

    # ---------------- window (sentence) level
    s_wtr, f_wtr = merged(win_train_sets)
    Xw, yw, gw, _, _, _ = window_arrays(s_wtr, f_wtr, win_names)
    rng = np.random.default_rng(seed)
    if len(yw) > max_windows:
        keep = np.sort(rng.choice(len(yw), max_windows, replace=False))
        Xw, yw, gw = Xw[keep], yw[keep], gw[keep]
    if verbose:
        print(f"Window level: {len(yw)} training windows ({yw.sum()} AI-centred), {len(win_names)} features", flush=True)
    win_ens = Ensemble("window", win_names, seed=seed, folds=folds).fit(Xw, yw, gw, verbose=verbose)
    Xwc, ywc, gwc, _, didx_c, cen_c = window_arrays(s_cal, f_cal, win_names)
    win_cal = win_ens.calibrate(Xwc, ywc, tuple(cfg["calibration"]["methods"]))
    if verbose:
        print(f"  calibration: selected {win_cal['selected']} | " + ", ".join(
            f"{k}: brier={v['brier']:.4f} ece={v['ece']:.4f}" for k, v in win_cal["results"].items()), flush=True)

    # ---------------- bands and share calibration (calibration split only)
    pwc = win_ens.predict_proba(Xwc)
    words_c = np.array([f_cal[d]["sentence_words"][c] for d, c in zip(didx_c, cen_c)], dtype=float)
    bands = derive_bands(pwc, ywc, np.maximum(words_c, 1.0), float(cfg["bands"]["target_human_flag_rate"]))
    raw_shares, true_shares = [], []
    for d in np.unique(didx_c):
        m = didx_c == d
        raw_shares.append(raw_share(pwc[m], words_c[m]))
        true_shares.append(float(s_cal[f_cal[d]["id"]]["ai_fraction"]))
    share_cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_shares, true_shares)

    fx_spec = FeatureExtractor(lang, tc["profile"], cfg).spec
    ds_version = dataset_version(list(tr_s.values()) + list(sets["calib"][0].values()))
    model_version = f"{lang}-{time.strftime('%Y%m%d')}-{ds_version}" + (f"-{tag}" if tag else "")
    hyper = {"folds": folds, "max_windows": max_windows, "include_nonnative": include_nonnative,
             "detectors": {n: d.kind for n, d in doc_ens.detectors.items()}, "window_sentences": fx_spec["window_sentences"],
             "stacker": "logistic regression, non-negative weights by pruning", "class_weight": "balanced"}
    training_info = {
        "dataset_version": ds_version, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed,
        "n_doc_train": int(len(yd)), "n_doc_calib": int(len(ydc)), "n_window_train": int(len(yw)),
        "n_window_calib": int(len(ywc)), "train_sets": train_sets, "hyperparameters": hyper,
        "doc_oof": {n: d.validation for n, d in doc_ens.detectors.items()},
        "window_oof": {n: d.validation for n, d in win_ens.detectors.items()},
        "doc_weights": doc_ens.weights, "window_weights": win_ens.weights,
        "doc_calibration": _strip_rel(doc_cal), "window_calibration": _strip_rel(win_cal),
        "generators_in_training": sorted({m["generator"] for m in s_tr.values() if m["label"] == 1}),
        "domains_in_training": sorted({m["domain"] for m in s_tr.values()}),
        "train_seconds": round(time.time() - t0, 1),
    }
    bundle = {
        "format": 1, "model_version": model_version, "feature_version": FEATURE_VERSION, "spec": fx_spec,
        "doc": doc_ens, "window": win_ens, "bands": bands, "share_calibrator": share_cal,
        "evidence_thresholds": dict(cfg["evidence"]["thresholds_words"]), "length_table": None,
        "evidence_derived": False, "training": training_info,
        "calibration_reports": {"document": doc_cal, "window": win_cal},
        "evaluation": None,
    }
    out_path = Path(out_path or MODELS_DIR / lang / "bundle.joblib")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path, compress=3)
    run = log_run("train", dataset_version=ds_version, feature_version=FEATURE_VERSION, model_version=model_version,
                  hyperparameters=hyper, seed=seed,
                  validation={"doc_oof": training_info["doc_oof"], "doc_calibration": training_info["doc_calibration"],
                              "window_calibration": training_info["window_calibration"], "bands": bands})
    if verbose:
        print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s; run log {run.name}")
        print("Learned document-level weights:", json.dumps({k: round(v, 3) for k, v in doc_ens.weights.items()}))
    return bundle


def _strip_rel(rep: dict) -> dict:
    return {"selected": rep["selected"],
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "reliability"} for k, v in rep["results"].items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=60000)
    ap.add_argument("--include-nonnative", action="store_true",
                    help="add half of the non-native human essays to training (false-positive experiment)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    train(args.lang, Path(args.data_dir) if args.data_dir else None, Path(args.out) if args.out else None,
          args.folds, args.max_windows, args.include_nonnative, args.tag)


if __name__ == "__main__":
    main()
