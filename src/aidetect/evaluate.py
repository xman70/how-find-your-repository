"""Evaluation pipeline. Every number in the evaluation report is produced here.

    python evaluate.py --lang en                 # full evaluation (+ derive length thresholds)
    python evaluate.py --lang en --quick         # skip the leave-one-out retraining studies

Studies
  1  held-out test set (document level) with group-bootstrap 95 % CIs, by domain and generator
  2  calibration on the test set: uncalibrated vs Platt vs isotonic (fitted on the calibration split)
  3  end-to-end inference on test documents + spliced hybrids: sentence level, AI share, 5-way hybrid
     classes, change-point detection, confidence validity
  4  length study on truncated test documents -> evidence thresholds and length-reliability table
  5  cross-generator generalisation (leave-one-generator-family-out retraining)
  6  cross-domain generalisation (leave-one-domain-out retraining)
  7  baseline model comparison (LR, RF, HGB, LightGBM, SVM, MLP)
  8  adversarial / robustness categories
  9  false-positive audit on held-out human populations (non-native, students, academic, legal)
  10 detector disagreement and global feature importance
  11 shortcut (dataset-artefact) audit: single-feature AUC within each domain
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from .calibration import CALIBRATORS, balanced_weights, brier, ece, reliability
from .config import DATA_DIR, MODELS_DIR, REPORTS_DIR, load_config
from .data import HYBRID_NAMES, truncate_words, count_words
from .features import featurize_corpus, to_matrix
from .metrics import binary_metrics, rate_ci, summarize
from .models import Ensemble, compare_baselines
from .train import doc_arrays, load_set, window_arrays
from .tracking import log_run

NAN = float("nan")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 5)
    if isinstance(o, np.integer):
        return int(o)
    return o


def derive_thresholds(rows: list[dict], criteria: dict) -> dict:
    """Smallest tested length from which every longer length meets the level's criteria
    (``min_auc`` and optionally ``max_fpr`` at the 0.5 threshold)."""
    out = {}
    for level in ("low", "moderate", "reliable"):
        c = criteria[level]

        def ok(r):
            return r["roc_auc"] >= c["min_auc"] and r["fpr"] <= c.get("max_fpr", 1.0)

        cands = [r["words"] for i, r in enumerate(rows) if all(ok(rr) for rr in rows[i:])]
        out[level] = int(min(cands)) if cands else None
    return out


def rederive(lang: str = "en", bundle_path=None, out_dir=None) -> dict:
    """Re-derive evidence thresholds from saved length-study measurements with the current config criteria."""
    cfg = load_config()
    out_dir = Path(out_dir or REPORTS_DIR / "evaluation" / lang)
    res_path = out_dir / "evaluation_results.json"
    res = json.loads(res_path.read_text(encoding="utf-8"))
    L = res["length"]
    if "initial_derivation" not in L:
        L["initial_derivation"] = {"criteria": L["criteria"], "derived_thresholds": L["derived_thresholds"]}
    L["criteria"] = cfg["evidence"]["criteria"]
    L["derived_thresholds"] = derive_thresholds(L["rows"], L["criteria"])
    ev = Evaluator.__new__(Evaluator)
    ev.bundle_path = Path(bundle_path or MODELS_DIR / lang / "bundle.joblib")
    ev.bundle = joblib.load(ev.bundle_path)
    ev.results = {"length": L}
    ev.verbose = True
    ev.write_back_thresholds()
    res["evidence_thresholds"] = ev.bundle["evidence_thresholds"]
    res_path.write_text(json.dumps(_clean(res), indent=2), encoding="utf-8")
    return L


class Evaluator:
    def __init__(self, lang="en", bundle_path=None, data_dir=None, seed=13, n_boot=500, jobs=4, verbose=True):
        self.cfg = load_config()
        self.lang = lang
        self.bundle_path = Path(bundle_path or MODELS_DIR / lang / "bundle.joblib")
        self.bundle = joblib.load(self.bundle_path)
        self.data_dir = Path(data_dir or DATA_DIR / "processed" / lang)
        self.seed = seed
        self.n_boot = n_boot
        self.jobs = jobs
        self.verbose = verbose
        self.doc_ens: Ensemble = self.bundle["doc"]
        self.win_ens: Ensemble = self.bundle["window"]
        self.results: dict = {"model_version": self.bundle["model_version"], "date": time.strftime("%Y-%m-%d %H:%M:%S")}

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _set(self, name):
        return load_set(self.data_dir, name)

    def _doc_pred(self, samples, feats):
        X, y, g, meta = doc_arrays(samples, feats, self.doc_ens.feature_names)
        return self.doc_ens.predict_proba(X) if len(y) else np.zeros(0), y, g, meta, X

    # ---------------------------------------------------------------- 1
    def conservative_threshold(self, target_fpr: float = 0.01) -> float:
        """Document threshold giving ``target_fpr`` on human documents of the calibration split."""
        if getattr(self, "_t_cons", None) is None:
            cs, cf = self._set("calib")
            pc, yc, *_ = self._doc_pred(cs, cf)
            self._t_cons = float(np.quantile(pc[yc == 0], 1 - target_fpr))
        return self._t_cons

    def study_test(self):
        self.log("[1] held-out test set")
        s, f = self._set("test")
        p, y, g, meta, X = self._doc_pred(s, f)
        t1 = self.conservative_threshold()
        out = {"overall": summarize(y, p, g, n_boot=self.n_boot, seed=self.seed),
               "conservative_operating_point": {"threshold": t1, "derivation": "1% FPR on calibration-split human documents",
                                                **summarize(y, p, g, threshold=t1, n_boot=self.n_boot, seed=self.seed)}}
        by_dom = {}
        for d in sorted({m["domain"] for m in meta}):
            m = np.array([mm["domain"] == d for mm in meta])
            by_dom[d] = binary_metrics(y[m], p[m])
            by_dom[d]["fpr_ci95"] = rate_ci(int(((p >= 0.5) & (y == 0) & m).sum()), int(((y == 0) & m).sum()))
        out["by_domain"] = by_dom
        by_gen = {}
        for gname in sorted({mm["generator"] for mm in meta if mm["label"] == 1}):
            m = np.array([mm["generator"] == gname for mm in meta])
            k = int((p[m] >= 0.5).sum())
            by_gen[gname] = {"n": int(m.sum()), "detection_rate": k / max(1, int(m.sum())),
                             "detection_rate_ci95": rate_ci(k, int(m.sum())), "mean_probability": float(p[m].mean())}
        out["ai_detection_by_generator"] = by_gen
        # hybrids (document target: AI share >= 0.5)
        hs, hf = self._set("hybrids_test")
        hp, hy, hg, hmeta, _ = self._doc_pred(hs, hf)
        out["hybrids_document_level"] = binary_metrics(hy, hp)
        self.results["test"] = out
        self._test_cache = (p, y, g, meta, X)

    # ---------------------------------------------------------------- 2
    def study_calibration(self):
        self.log("[2] calibration comparison on the test set")
        res = {}
        for level, ens, split_names in (("document", self.doc_ens, ("calib", "hybrids_calib")),
                                        ("window", self.win_ens, ("calib", "hybrids_calib"))):
            cs, cf = {}, []
            for n in split_names:
                a, b = self._set(n)
                cs.update(a)
                cf.extend(b)
            ts, tf = {}, []
            for n in ("test", "hybrids_test"):
                a, b = self._set(n)
                ts.update(a)
                tf.extend(b)
            if level == "document":
                Xc, yc, _, _ = doc_arrays(cs, cf, ens.feature_names)
                Xt, yt, _, _ = doc_arrays(ts, tf, ens.feature_names)
            else:
                Xc, yc, *_ = window_arrays(cs, cf, ens.feature_names)
                Xt, yt, *_ = window_arrays(ts, tf, ens.feature_names)
            rc, rt = ens.predict_raw(Xc), ens.predict_raw(Xt)
            w = balanced_weights(yt)
            entry = {"selected_on_calibration_split": ens.calibration_report["selected"],
                     "cv_on_calibration_split": {k: {kk: vv for kk, vv in v.items() if kk != "reliability"}
                                                 for k, v in ens.calibration_report["results"].items()},
                     "test": {}}
            for m in ("none", "platt", "isotonic"):
                cal = CALIBRATORS[m]().fit(rc, yc, balanced_weights(yc))
                pt = cal.transform(rt)
                entry["test"][m] = {"brier": brier(yt, pt, w), "ece": ece(yt, pt, 10, w),
                                    "reliability": reliability(yt, pt, 10, w), "n": int(len(yt))}
            res[level] = entry
        self.results["calibration"] = res

    # ---------------------------------------------------------------- 3
    def study_end_to_end(self, max_pure=300):
        self.log("[3] end-to-end inference (sentence level, shares, hybrid classes, change points)")
        from .predict import Analyzer

        an = Analyzer(config=self.cfg, models_dir=self.bundle_path.parent.parent)
        an._bundles[self.lang] = self.bundle
        an.cfg = json.loads(json.dumps(self.cfg))
        an.cfg["changepoint"]["permutations"] = 100
        hs, _ = self._set("hybrids_test")
        ts, _ = self._set("test")
        rng = np.random.default_rng(self.seed)
        pure = [s for s in ts.values() if s.get("author_type") in ("human", "ai")]
        pure = [pure[i] for i in rng.permutation(len(pure))[:max_pure]]
        docs = list(hs.values()) + pure
        sent_p, sent_y, share_true, share_est, share_band, cls_true, cls_pred = [], [], [], [], [], [], []
        confs, correct, cps_tp, cps_fn, cps_fp, pure_alarm, pure_n = [], [], 0, 0, 0, 0, 0
        from .features import sentence_labels

        for i, smp in enumerate(docs):
            r = an.analyze(smp["text"], title=smp["id"], language=self.lang)
            if r["status"] != "ok":
                continue
            fx = an._extractors[self.lang]
            doc = fx.parse(smp["text"], normalized=True)
            labels = sentence_labels(doc, smp)
            if len(labels) != len(r["sentences"]):
                continue
            sp = np.array([x["probability"] for x in r["sentences"]])
            sent_p.extend(sp.tolist())
            sent_y.extend(labels.tolist())
            res = r["result"]
            words = np.array([x["words"] for x in r["sentences"]], float)
            share_true.append(float(np.sum(labels * words) / words.sum()))
            share_est.append(res["estimated_ai_share"])
            share_band.append(res["ai_associated_share"])
            from .data import hybrid_class_for

            cls_true.append(hybrid_class_for(share_true[-1], self.cfg))
            cls_pred.append(res["hybrid"]["code"])
            confs.append(res["confidence"])
            correct.append(int((res["document_probability"] >= 0.5) == (share_true[-1] >= 0.5)))
            true_b = [j for j in range(1, len(labels)) if labels[j] != labels[j - 1]]
            pred_b = [c["boundary_before_sentence"] for c in r["transitions"]["changepoints"]]
            if smp.get("segments"):
                matched = set()
                for tb in true_b:
                    hit = [pb for pb in pred_b if abs(pb - tb) <= 2 and pb not in matched]
                    if hit:
                        cps_tp += 1
                        matched.add(hit[0])
                    else:
                        cps_fn += 1
                cps_fp += len([pb for pb in pred_b if pb not in matched])
            else:
                pure_n += 1
                pure_alarm += int(len(pred_b) > 0)
            if (i + 1) % 100 == 0:
                self.log(f"    {i + 1}/{len(docs)} documents")
        sent_p, sent_y = np.array(sent_p), np.array(sent_y).astype(int)
        cm = np.zeros((4, 5), dtype=int)
        for t, pr in zip(cls_true, cls_pred):
            cm[t, pr] += 1
        decided = [(t, pr) for t, pr in zip(cls_true, cls_pred) if pr != 4]
        conf_arr, corr_arr = np.array(confs), np.array(correct)
        conf_bins = {}
        for name, lo, hi in (("Low", 0, 0.4), ("Medium", 0.4, 0.7), ("High", 0.7, 1.01)):
            m = (conf_arr >= lo) & (conf_arr < hi)
            conf_bins[name] = {"n": int(m.sum()), "accuracy": float(corr_arr[m].mean()) if m.any() else None}
        prec = cps_tp / max(1, cps_tp + cps_fp)
        rec = cps_tp / max(1, cps_tp + cps_fn)
        self.results["end_to_end"] = {
            "n_documents": len(share_true), "n_sentences": int(len(sent_y)),
            "sentence_level": {**binary_metrics(sent_y, sent_p),
                               "reliability": reliability(sent_y, sent_p, 10, balanced_weights(sent_y))},
            "ai_share": {"mae_estimated_share": float(np.mean(np.abs(np.array(share_est) - np.array(share_true)))),
                         "mae_band_share": float(np.mean(np.abs(np.array(share_band) - np.array(share_true)))),
                         "correlation_estimated": float(np.corrcoef(share_est, share_true)[0, 1])},
            "hybrid_classes": {"labels_true": [HYBRID_NAMES[i] for i in range(4)],
                               "labels_pred": [HYBRID_NAMES[i] for i in range(5)], "confusion": cm.tolist(),
                               "accuracy_when_decided": (float(np.mean([t == pr for t, pr in decided])) if decided else None),
                               "within_one_class_when_decided": (float(np.mean([abs(t - pr) <= 1 for t, pr in decided]))
                                                                 if decided else None),
                               "unknown_rate": float(np.mean(np.array(cls_pred) == 4))},
            "changepoints": {"precision_within_2_sentences": prec, "recall_within_2_sentences": rec,
                             "true_boundaries": cps_tp + cps_fn,
                             "false_alarm_rate_pure_documents": pure_alarm / max(1, pure_n), "n_pure": pure_n},
            "confidence_validity": conf_bins,
        }

    # ---------------------------------------------------------------- 4
    def study_length(self, lengths=(25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 700), max_docs=700):
        self.log("[4] length study (truncated held-out documents)")
        ts, _ = self._set("test")
        pure = [s for s in ts.values() if s.get("author_type") in ("human", "ai")]
        rng = np.random.default_rng(self.seed)
        rows = []
        for L in lengths:
            elig = [s for s in pure if s["n_words"] >= L]
            if len(elig) < 60 or len({s["label"] for s in elig}) < 2:
                continue
            pick = [elig[i] for i in rng.permutation(len(elig))[:max_docs]]
            items = []
            for s in pick:
                t = truncate_words(s["text"], L)
                items.append(dict(s, text=t, id=s["id"] + f":L{L}", n_words=count_words(t)))
            feats = [r for r in featurize_corpus(items, self.lang, self.bundle["spec"]["profile"], self.cfg,
                                                 n_jobs=self.jobs, progress=False) if "error" not in r]
            smap = {x["id"]: x for x in items}
            p, y, g, meta, _ = self._doc_pred(smap, feats)
            m = binary_metrics(y, p)
            rows.append({"words": L, "n": int(len(y)), "roc_auc": m["roc_auc"], "fpr": m["fpr"], "fnr": m["fnr"],
                         "accuracy": m["accuracy"], "mean_words": float(np.mean([x["n_words"] for x in meta]))})
            self.log(f"    {L:4d} words: n={len(y)} AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f}")
        crit = self.cfg["evidence"]["criteria"]
        thresholds = derive_thresholds(rows, crit)
        best = max((r["roc_auc"] for r in rows), default=1.0)
        table = [[0, 0.0]] + [[r["words"], float(np.clip((r["roc_auc"] - 0.5) / max(best - 0.5, 1e-6), 0, 1))] for r in rows]
        for i in range(1, len(table)):  # monotone non-decreasing reliability
            table[i][1] = max(table[i][1], table[i - 1][1])
        auc_only = {"low": {"min_auc": 0.75}, "moderate": {"min_auc": 0.85}, "reliable": {"min_auc": 0.92}}
        self.results["length"] = {"rows": rows, "derived_thresholds": thresholds, "criteria": crit,
                                  "initial_derivation": {"criteria": auc_only,
                                                         "derived_thresholds": derive_thresholds(rows, auc_only)},
                                  "length_reliability_table": table}

    def write_back_thresholds(self):
        L = self.results.get("length")
        if not L:
            return
        th = dict(self.bundle["evidence_thresholds"])
        derived = L["derived_thresholds"]
        max_len = max(r["words"] for r in L["rows"])
        for k in ("low", "moderate", "reliable"):
            th[k] = derived[k] if derived[k] is not None else max_len + 1
        th["low"] = min(th["low"], th["moderate"], th["reliable"])
        th["moderate"] = max(th["moderate"], th["low"])
        th["reliable"] = max(th["reliable"], th["moderate"])
        self.bundle["evidence_thresholds"] = th
        self.bundle["length_table"] = L["length_reliability_table"]
        self.bundle["evidence_derived"] = True
        joblib.dump(self.bundle, self.bundle_path, compress=3)
        self.log(f"    evidence thresholds written to bundle: {th}")

    # ---------------------------------------------------------------- 5/6 retraining studies
    def _train_sets(self) -> tuple[str, ...]:
        hp = self.bundle["training"].get("hyperparameters", {})
        return ("train", "train_augmented") + (("nonnative_train",) if hp.get("include_nonnative") else ())

    def _retrain_eval(self, exclude_fn, test_fn, folds=3):
        sets = {n: self._set(n) for n in self._train_sets() + ("calib", "test")}
        names = self.doc_ens.feature_names

        def arrays(set_names, keep):
            s, f = {}, []
            for n in set_names:
                a, b = sets[n]
                s.update(a)
                f.extend(r for r in b if keep(a[r["id"]]))
            return doc_arrays(s, f, names)

        Xtr, ytr, gtr, _ = arrays(self._train_sets(), lambda m: not exclude_fn(m))
        Xc, yc, _, _ = arrays(("calib",), lambda m: not exclude_fn(m))
        Xt, yt, gt, mt = arrays(("test",), test_fn)
        if len(set(yt)) < 2 or len(set(ytr)) < 2:
            return None
        ens = Ensemble("document", names, seed=self.seed, folds=folds).fit(Xtr, ytr, gtr, verbose=False)
        ens.calibrate(Xc, yc)
        p = ens.predict_proba(Xt)
        ref = self.doc_ens.predict_proba(Xt)
        return {"held_out": summarize(yt, p, gt, n_boot=200, seed=self.seed),
                "in_distribution_reference": binary_metrics(yt, ref), "n_train": int(len(ytr)), "n_test": int(len(yt))}

    def study_cross_generator(self):
        self.log("[5] cross-generator generalisation (leave-one-family-out)")
        ts, _ = self._set("test")
        fams = sorted({m["generator_family"] for m in ts.values() if m["label"] == 1})
        out = self.results.setdefault("cross_generator", {"_partial": True})
        for fam in fams:
            if fam in out:
                continue
            t0 = time.time()
            r = self._retrain_eval(lambda m, f=fam: m["label"] == 1 and m["generator_family"] == f,
                                   lambda m, f=fam: m["label"] == 0 or m["generator_family"] == f)
            if r:
                out[fam] = r
                self.log(f"    held out {fam:11s}: AUC={r['held_out']['roc_auc']:.3f} TPR={r['held_out']['recall']:.3f} "
                         f"FPR={r['held_out']['fpr']:.3f} (in-distribution AUC {r['in_distribution_reference']['roc_auc']:.3f}) "
                         f"[{time.time() - t0:.0f}s]")
                self.checkpoint()

    def study_cross_domain(self):
        self.log("[6] cross-domain generalisation (leave-one-domain-out)")
        ts, _ = self._set("test")
        out = self.results.setdefault("cross_domain", {"_partial": True})
        for dom in sorted({m["domain"] for m in ts.values()}):
            if dom in out:
                continue
            t0 = time.time()
            r = self._retrain_eval(lambda m, d=dom: m["domain"] == d, lambda m, d=dom: m["domain"] == d)
            if r:
                out[dom] = r
                self.log(f"    held out {dom:10s}: AUC={r['held_out']['roc_auc']:.3f} FPR={r['held_out']['fpr']:.3f} "
                         f"FNR={r['held_out']['fnr']:.3f} (in-distribution AUC {r['in_distribution_reference']['roc_auc']:.3f}) "
                         f"[{time.time() - t0:.0f}s]")
                self.checkpoint()

    # ---------------------------------------------------------------- 7
    def study_baselines(self):
        self.log("[7] baseline comparison")
        names = self.doc_ens.feature_names
        tr = [self._set(n) for n in self._train_sets()]
        s, f = {}, []
        for a, b in tr:
            s.update(a)
            f.extend(b)
        Xtr, ytr, _, _ = doc_arrays(s, f, names)
        cs, cf = self._set("calib")
        Xc, yc, _, _ = doc_arrays(cs, cf, names)
        p, yt, g, meta, Xt = self._test_cache
        res = compare_baselines(Xtr, ytr, Xt, yt, Xc, yc, seed=self.seed)
        res["stacked_ensemble (this system)"] = binary_metrics(yt, p)
        self.results["baselines"] = res

    # ---------------------------------------------------------------- 8
    def study_adversarial(self):
        self.log("[8] adversarial and robustness categories")
        s, f = self._set("adversarial")
        p, y, g, meta, X = self._doc_pred(s, f)
        test_p, test_y, _, test_meta, _ = self._test_cache
        base_by_group = defaultdict(list)
        for pp, mm in zip(test_p, test_meta):
            base_by_group[mm["group_id"]].append((pp, mm["label"]))
        out = {}
        for es in sorted({m.get("eval_set", "?") for m in meta}):
            m = np.array([mm.get("eval_set") == es for mm in meta])
            yy, pp = y[m], p[m]
            e = {"n": int(m.sum()), "n_ai": int(yy.sum()), "n_human": int((yy == 0).sum())}
            if yy.sum():
                k = int((pp[yy == 1] >= 0.5).sum())
                e["ai_detection_rate"] = k / int(yy.sum())
                e["ai_detection_rate_ci95"] = rate_ci(k, int(yy.sum()))
                e["ai_mean_probability"] = float(pp[yy == 1].mean())
            if (yy == 0).sum():
                k = int((pp[yy == 0] >= 0.5).sum())
                e["human_flag_rate"] = k / int((yy == 0).sum())
                e["human_flag_rate_ci95"] = rate_ci(k, int((yy == 0).sum()))
            if yy.sum() and (yy == 0).sum():
                e["roc_auc"] = float(roc_auc_score(yy, pp))
            # reference: unperturbed versions of the same documents in the test split, when available
            refs = [v for mm in np.array(meta, dtype=object)[m] for v in base_by_group.get(mm["group_id"], [])]
            if refs:
                rp = np.array([r[0] for r in refs])
                ry = np.array([r[1] for r in refs])
                if ry.sum():
                    e["reference_unperturbed_ai_detection_rate"] = float((rp[ry == 1] >= 0.5).mean())
                if (ry == 0).sum():
                    e["reference_unperturbed_human_flag_rate"] = float((rp[ry == 0] >= 0.5).mean())
            out[es] = e
        self.results["adversarial"] = out

    # ---------------------------------------------------------------- 9
    def study_fp_audit(self, exclude_nonnative_train_half=True):
        self.log("[9] false-positive audit on held-out human populations")
        s, f = self._set("fp_audit")
        if exclude_nonnative_train_half:
            f = [r for r in f if not s[r["id"]].get("nonnative_train_half")]
        p, y, g, meta, X = self._doc_pred(s, f)
        # sentence-level flagged share for each document
        Xw, yw, gw, frw, didx, cen = window_arrays(s, f, self.win_ens.feature_names)
        pw = self.win_ens.predict_proba(Xw) if len(yw) else np.zeros(0)
        t_hi = self.bundle["bands"]["t_high"]
        share = np.zeros(len(f))
        for d in range(len(f)):
            mk = didx == d
            if mk.any():
                words = np.array([f[d]["sentence_words"][c] for c in cen[mk]], float)
                share[d] = float(np.sum(words * (pw[mk] >= t_hi)) / max(words.sum(), 1))
        test_p, test_y, _, test_meta, _ = self._test_cache
        ref_fpr = float((test_p[test_y == 0] >= 0.5).mean())
        groups = {}
        for es in sorted({m.get("eval_set", "?") for m in meta}):
            m = np.array([mm.get("eval_set") == es for mm in meta])
            k = int((p[m] >= 0.5).sum())
            n = int(m.sum())
            k1 = int((p[m] >= self.conservative_threshold()).sum())
            groups[es] = {"fpr_conservative": k1 / n, "fpr_conservative_ci95": rate_ci(k1, n),
                          "n": n, "writer_group": Counter(mm["writer_group"] for mm in np.array(meta, dtype=object)[m]).most_common(1)[0][0],
                          "genre": meta[int(np.flatnonzero(m)[0])]["genre"], "fpr": k / n, "fpr_ci95": rate_ci(k, n),
                          "mean_probability": float(p[m].mean()),
                          "mean_ai_associated_share": float(share[m].mean()),
                          "share_docs_with_ai_associated_majority": float((share[m] >= 0.5).mean()),
                          "fpr_ratio_vs_in_distribution": (k / n) / ref_fpr if ref_fpr > 0 else None,
                          "median_words": float(np.median([mm["n_words"] for mm in np.array(meta, dtype=object)[m]]))}
        # in-distribution human test groups that match the audit categories
        for dom, label in (("arxiv", "test_arxiv_scientific"), ("peerread", "test_peerread_academic_reviews"),
                           ("wikipedia", "test_wikipedia_highly_edited"), ("reuter", "test_reuters_professional_news"),
                           ("essay", "test_student_essays")):
            m = np.array([(mm["domain"] == dom and mm["label"] == 0) for mm in test_meta])
            if m.any():
                k = int((test_p[m] >= 0.5).sum())
                k1 = int((test_p[m] >= self.conservative_threshold()).sum())
                groups[label] = {"fpr_conservative": k1 / int(m.sum()), "fpr_conservative_ci95": rate_ci(k1, int(m.sum())),
                                 "n": int(m.sum()), "writer_group": "unknown", "genre": test_meta[int(np.flatnonzero(m)[0])]["genre"],
                                 "fpr": k / int(m.sum()), "fpr_ci95": rate_ci(k, int(m.sum())),
                                 "mean_probability": float(test_p[m].mean()), "in_distribution": True}
        self.results["fp_audit"] = {"reference_in_distribution_fpr": ref_fpr, "groups": groups,
                                    "conservative_threshold": self.conservative_threshold(),
                                    "reference_in_distribution_fpr_conservative": float(
                                        (test_p[test_y == 0] >= self.conservative_threshold()).mean()),
                                    "not_tested": ["dyslexic writers (no accessible public corpus in this environment)"],
                                    "excluded_nonnative_training_half": exclude_nonnative_train_half}

    # ---------------------------------------------------------------- 10
    def study_detectors(self):
        self.log("[10] detector disagreement and feature importance")
        p, y, g, meta, X = self._test_cache
        outs = self.doc_ens.detector_outputs(X)
        det = {}
        for name, o in outs.items():
            det[name] = {"test_roc_auc": float(roc_auc_score(y, o["prob"])), "mean_committee_std": float(o["std"].mean()),
                         "weight": self.doc_ens.weights.get(name, 0.0)}
        names = list(outs)
        P = np.vstack([outs[n]["prob"] for n in names])
        corr = np.corrcoef(P)
        spread = P.std(axis=0)
        correct = ((p >= 0.5) == (y == 1)).astype(float)
        hi = spread >= np.median(spread)
        from sklearn.inspection import permutation_importance

        stat = self.doc_ens.detectors["statistical"]
        rng = np.random.default_rng(self.seed)
        sub = rng.permutation(len(y))[:1500]
        pi = permutation_importance(stat.models[0], X[sub][:, stat.cols], y[sub], scoring="roc_auc", n_repeats=3,
                                    random_state=self.seed)
        order = np.argsort(-pi.importances_mean)[:20]
        self.results["detectors"] = {
            "per_detector": det, "correlation": {"names": names, "matrix": corr.tolist()},
            "accuracy_when_detectors_agree": float(correct[~hi].mean()),
            "accuracy_when_detectors_disagree": float(correct[hi].mean()),
            "statistical_model_permutation_importance_top20": [
                {"feature": stat.feature_names[j], "auc_drop": float(pi.importances_mean[j])} for j in order],
        }

    # ---------------------------------------------------------------- 11
    def study_shortcuts(self, threshold=0.9):
        self.log("[11] shortcut / artefact audit")
        s, f = self._set("train")
        X, y, g, meta = doc_arrays(s, f, self.doc_ens.feature_names)
        flagged = []
        for dom in sorted({m["domain"] for m in meta}):
            m = np.array([mm["domain"] == dom for mm in meta])
            if len(set(y[m])) < 2:
                continue
            for j, name in enumerate(self.doc_ens.feature_names):
                col = X[m, j]
                ok = ~np.isnan(col)
                if ok.sum() < 50 or len(set(y[m][ok])) < 2:
                    continue
                auc = roc_auc_score(y[m][ok], col[ok])
                strength = max(auc, 1 - auc)
                if strength >= threshold:
                    flagged.append({"domain": dom, "feature": name, "single_feature_auc": float(strength),
                                    "direction": "higher in AI" if auc >= 0.5 else "higher in human"})
        flagged.sort(key=lambda r: -r["single_feature_auc"])
        self.results["shortcut_audit"] = {"threshold": threshold, "flagged": flagged[:60], "n_flagged": len(flagged),
                                          "note": "A single feature separating classes almost perfectly inside one "
                                                  "domain can indicate a collection artefact; each flagged item was "
                                                  "reviewed (see evaluation report)."}

    # ---------------------------------------------------------------- run
    def checkpoint(self):
        if getattr(self, "checkpoint_path", None):
            self.checkpoint_path.write_text(json.dumps(_clean(self.results), indent=2), encoding="utf-8")

    def run(self, quick=False, derive=True):
        """Runs every study, checkpointing after each one so an interrupted run can resume."""
        t0 = time.time()
        self.study_test()  # always recomputed: later studies reuse its predictions
        steps = [("calibration", self.study_calibration), ("detectors", self.study_detectors),
                 ("adversarial", self.study_adversarial), ("fp_audit", self.study_fp_audit),
                 ("shortcut_audit", self.study_shortcuts), ("baselines", self.study_baselines)]
        if derive:
            steps.append(("length", lambda: (self.study_length(), self.write_back_thresholds())))
        steps.append(("end_to_end", self.study_end_to_end))
        if not quick:
            steps += [("cross_generator", self.study_cross_generator), ("cross_domain", self.study_cross_domain)]
        for key, fn in steps:
            if key in self.results and not self.results[key].get("_partial", False):
                self.log(f"[resume] {key} already done")
                continue
            fn()
            self.results[key].pop("_partial", None)
            self.checkpoint()
        self.results["seconds"] = round(time.time() - t0, 1)
        self.results["training"] = self.bundle["training"]
        self.results["bands"] = self.bundle["bands"]
        self.results["evidence_thresholds"] = self.bundle["evidence_thresholds"]
        return self.results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None, help="output directory (default reports/evaluation/<lang>)")
    ap.add_argument("--quick", action="store_true", help="skip leave-one-out retraining studies")
    ap.add_argument("--no-derive", action="store_true", help="do not derive/write length thresholds")
    ap.add_argument("--only", default=None, help="comma-separated study names, e.g. fp_audit,test")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    ap.add_argument("--rederive-thresholds", action="store_true",
                    help="only re-derive evidence thresholds from saved length-study results")
    args = ap.parse_args(argv)
    if args.rederive_thresholds:
        L = rederive(args.lang, args.bundle, args.out)
        print("Derived thresholds:", L["derived_thresholds"])
        return
    ev = Evaluator(args.lang, args.bundle, args.data_dir, jobs=args.jobs)
    out_dir = Path(args.out or REPORTS_DIR / "evaluation" / args.lang)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.only:
        ev.study_test()
        for name in args.only.split(","):
            if name != "test":
                getattr(ev, f"study_{name}")()
        res = ev.results
    else:
        ev.checkpoint_path = out_dir / "evaluation_checkpoint.json"
        if ev.checkpoint_path.exists() and not args.fresh:
            prev = json.loads(ev.checkpoint_path.read_text(encoding="utf-8"))
            if prev.get("model_version") == ev.bundle["model_version"]:
                ev.results.update({k: v for k, v in prev.items() if k not in ("date", "model_version")})
                print(f"Resuming from checkpoint: {sorted(k for k in prev if isinstance(prev[k], dict))}")
        res = ev.run(quick=args.quick, derive=not args.no_derive)
    res = _clean(res)
    fname = "evaluation_results.json" if not args.only else f"evaluation_{args.only.replace(',', '_')}.json"
    (out_dir / fname).write_text(json.dumps(res, indent=2), encoding="utf-8")
    t = res.get("test", {}).get("overall", {})
    log_run("evaluate", dataset_version=ev.bundle["training"]["dataset_version"], feature_version=ev.bundle["feature_version"],
            model_version=ev.bundle["model_version"], hyperparameters={"quick": args.quick, "only": args.only},
            seed=ev.seed, test={k: t.get(k) for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr")})
    print(f"Results written to {out_dir / fname}")


if __name__ == "__main__":
    main()
