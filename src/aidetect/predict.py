"""Inference pipeline: text -> complete probabilistic analysis.

    python predict.py path/to/file.docx --report out.html

The result dictionary contains document-, paragraph- and sentence-level
calibrated probabilities, the AI/uncertain/human shares, confidence (separate
from likelihood), the five-way hybrid category, individual detector outputs
(model disagreement), explanations, statistics with reference percentiles,
potential style transitions, series for the graphs, limitations and the
mandatory disclaimer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import DISCLAIMER, INTERPRETATION, PRIVACY_NOTE, SIGNAL_CAVEAT, __version__
from . import confidence as conf_mod
from .config import MODELS_DIR, load_config
from .data import HYBRID_NAMES
from .device import resolve_device
from .calibration import logit
from .explain import ablation_contributions, factors_for_row, family_contributions, summary_text
from .features import FeatureExtractor, to_matrix
from .hybrid import classify
from .models import DETECTOR_LABELS, percentile_in
from .preprocessing import (ModelMissingError, PreprocessingError, count_words, detect_language, normalize_text)
from .signals import consistency
from .signals.textstats import mattr

MIN_WORDS_HARD = 30  # below this nothing is computed at all

LIMITATIONS = [
    "AI-text detection is probabilistic and produces both false positives (human text flagged) and false negatives "
    "(AI text missed).",
    "Probabilities are calibrated on held-out data with equal numbers of human and AI documents (a 50% prior). The "
    "share of AI text in your context changes how a given score should be read.",
    "Performance drops on text from AI models, domains, languages and writing populations that differ from the "
    "training data; see the evaluation report for measured cross-model and cross-domain results.",
    "Paraphrasing, translation, heavy human editing and short texts reduce detection reliability.",
    "Formulaic genres (legal, scientific, technical) and highly polished human writing can resemble AI-generated "
    "text statistically.",
    "Spelling or grammar mistakes, dialect, nationality and writing ability are never used as evidence of AI "
    "authorship, but non-native writing can still differ statistically from the training data; check the "
    "false-positive audit in the evaluation report.",
    "Sentence-level scores use a 5-sentence context window and are less reliable than document-level scores.",
    "Detected style transitions indicate a statistical change in writing characteristics, not a change of author.",
]


class Analyzer:
    def __init__(self, config: dict | None = None, device: str | None = None, models_dir: Path | None = None):
        self.cfg = config or load_config()
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.device_request = device or self.cfg["runtime"]["device"]
        self.device, self.device_note = resolve_device(self.device_request)
        self._bundles: dict[str, dict] = {}
        self._extractors: dict[str, FeatureExtractor] = {}

    # ------------------------------------------------------------------ resources
    def available_languages(self) -> list[str]:
        return sorted(p.parent.name for p in self.models_dir.glob("*/bundle.joblib"))

    def bundle(self, lang: str) -> dict:
        if lang not in self._bundles:
            path = self.models_dir / lang / "bundle.joblib"
            if not path.exists():
                raise ModelMissingError(
                    f"No trained model for language '{lang}' was found at {path}. Train one with 'python train.py --lang "
                    f"{lang}' after building the dataset, or reinstall the shipped model.")
            import joblib

            try:
                self._bundles[lang] = joblib.load(path)
            except Exception as exc:  # noqa: BLE001
                raise ModelMissingError(f"The model file {path} could not be loaded ({exc}). Retrain or reinstall it.") from exc
        return self._bundles[lang]

    def extractor(self, lang: str, bundle: dict) -> FeatureExtractor:
        if lang not in self._extractors:
            spec = bundle["spec"]
            fx = FeatureExtractor(lang, spec["profile"], self.cfg, self.device)
            if fx.spec["feature_version"] != spec["feature_version"]:
                raise ModelMissingError(
                    f"The installed model was trained with feature version {spec['feature_version']} but the code "
                    f"computes version {fx.spec['feature_version']}. Retrain the model.")
            self._extractors[lang] = fx
        return self._extractors[lang]

    # ------------------------------------------------------------------ main entry
    def analyze(self, text: str, title: str | None = None, language: str | None = None) -> dict:
        t0 = time.time()
        warnings: list[str] = []
        if self.device_note:
            warnings.append(self.device_note)
        norm = normalize_text(text or "")
        n_words = count_words(norm)
        base = {"meta": self._meta(title, norm, n_words), "disclaimer": DISCLAIMER, "limitations": LIMITATIONS,
                "privacy": PRIVACY_NOTE, "warnings": warnings}
        if n_words < MIN_WORDS_HARD:
            return {**base, "status": "insufficient_text",
                    "message": f"The text has {n_words} words. At least {MIN_WORDS_HARD} words are needed to compute "
                               "any statistics, and much more for a meaningful estimate."}
        guess = detect_language(norm, tuple(self.cfg["languages"]["supported"]))
        lang = language or guess.code
        base["meta"].update(language=lang, language_confidence=round(guess.confidence, 3))
        if not language and not guess.supported:
            return {**base, "status": "unsupported_language",
                    "message": guess.detail or "Unsupported language.",
                    "supported_languages": self.cfg["languages"]["supported"]}
        try:
            bundle = self.bundle(lang)
            fx = self.extractor(lang, bundle)
        except ModelMissingError as exc:
            status = "model_missing" if lang in self.cfg["languages"]["supported"] else "unsupported_language"
            if lang == "el" and status == "model_missing":
                exc = ModelMissingError(
                    "Greek is supported by the feature pipeline, but no Greek model has been trained and validated. "
                    "Applying the English model to Greek text would be scientifically invalid, so no result is "
                    "produced. See README: 'Greek support'.")
            return {**base, "status": status, "message": str(exc)}
        try:
            doc = fx.parse(norm, normalized=True, max_words=int(self.cfg["runtime"]["max_words"]))
        except PreprocessingError as exc:
            return {**base, "status": "error", "message": str(exc)}
        if doc.truncated:
            warnings.append(f"The document was truncated to about {self.cfg['runtime']['max_words']} words for analysis.")
        if len(doc.sentences) < 2:
            return {**base, "status": "insufficient_text", "message": "At least two sentences are needed."}
        from threadpoolctl import threadpool_limits

        # Small prediction batches are fastest single-threaded, and OpenMP spin-waiting makes multi-threaded
        # prediction extremely slow when other programs keep the CPU busy.
        with threadpool_limits(limits=int(self.cfg["runtime"].get("n_threads", 1)), user_api="openmp"):
            res = self._analyze_doc(doc, bundle, fx, base, warnings)
        res["meta"]["processing_seconds"] = round(time.time() - t0, 2)
        return res

    # ------------------------------------------------------------------ core
    def _meta(self, title, norm, n_words):
        return {"title": title or "Untitled document", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "word_count": n_words, "character_count": len(norm), "app_version": __version__,
                "device": self.device}

    def _analyze_doc(self, doc, bundle, fx, base, warnings) -> dict:
        cfg = self.cfg
        doc_ens, win_ens = bundle["doc"], bundle["window"]
        n = len(doc.sentences)
        words = np.array([max(s.c("n_words"), 1.0) for s in doc.sentences])
        n_words = int(words.sum())

        # document level
        dfeat = fx.document_features(doc)
        Xd = to_matrix([dfeat], doc_ens.feature_names)
        needs_text = bool(getattr(doc_ens, "text_detectors", None))
        dd = doc_ens.predict_detail(Xd, [doc.text] if needs_text else None)
        doc_prob = float(dd["prob"][0])
        det_doc = {k: {"prob": float(v["prob"][0]), "std": float(v["std"][0]), "confidence": float(v["confidence"][0])}
                   for k, v in dd["detectors"].items()}

        # sentence level (centred windows)
        wrows = fx.window_features(doc)
        Xw = to_matrix([f for _, _, f in wrows], win_ens.feature_names)
        win_texts = ([" ".join(doc.sentences[j].text for j in idx) for _, idx, _ in wrows]
                     if getattr(win_ens, "text_detectors", None) else None)
        wd = win_ens.predict_detail(Xw, win_texts)
        sp = wd["prob"]
        win_words = np.array([words[idx].sum() for _, idx, _ in wrows])

        bands = bundle["bands"]
        t_hi, t_lo = bands["t_high"], bands["t_low"]
        ai_mask, hu_mask = sp >= t_hi, sp <= t_lo
        share_ai = float(words[ai_mask].sum() / n_words)
        share_hu = float(words[hu_mask].sum() / n_words)
        share_un = max(0.0, 1.0 - share_ai - share_hu)
        raw_share = float(np.sum(sp * words) / n_words)
        est_share = float(bundle["share_calibrator"].predict([raw_share])[0]) if bundle.get("share_calibrator") else raw_share

        # confidence (separate from likelihood)
        probs = {k: v["prob"] for k, v in det_doc.items()}
        stds = {k: v["std"] for k, v in det_doc.items()}
        cov = conf_mod.coverage(Xd[0], doc_ens.feature_names, doc_ens.population)
        conf = conf_mod.overall(n_words, probs, doc_ens.weights, stds, cov, bundle.get("length_table"),
                                cfg["confidence"]["levels"])
        evidence = conf_mod.evidence_level(n_words, bundle.get("evidence_thresholds") or cfg["evidence"]["thresholds_words"])
        hybrid = classify(est_share, conf["value"], evidence, cfg)
        if evidence == "insufficient":
            warnings.append("Insufficient evidence: the text is shorter than the length at which the model was "
                            "validated as informative. Treat all numbers as unreliable.")
        if cov < 0.85:
            warnings.append(f"Only {cov:.0%} of the measured characteristics fall inside the range seen in training "
                            "data; the text may be unlike the training material (domain, genre or format).")

        # explanations (baseline-replacement attributions on the deployed ensemble)
        tl_doc = {k: logit(dd["detectors"][k]["prob"]) for k in getattr(doc_ens, "text_detectors", {}) if k in doc_ens.stack_names}
        cd = ablation_contributions(doc_ens, Xd, tl_doc)[0]
        doc_factors = factors_for_row(doc_ens, Xd[0], cd, top_k=5)
        doc_families = family_contributions(doc_ens, cd)
        tl_win = {k: logit(wd["detectors"][k]["prob"]) for k in getattr(win_ens, "text_detectors", {}) if k in win_ens.stack_names}
        cw = ablation_contributions(win_ens, Xw, tl_win)
        det_win_probs = {k: v["prob"] for k, v in wd["detectors"].items()}
        level_names = cfg["heatmap"]["levels"]

        sentences = []
        for i, s in enumerate(doc.sentences):
            p = float(sp[i])
            band = "ai" if p >= t_hi else ("human" if p <= t_lo else "uncertain")
            sw = {k: float(v[i]) for k, v in det_win_probs.items()}
            ag = conf_mod.agreement(sw, win_ens.weights)
            f = factors_for_row(win_ens, Xw[i], cw[i], top_k=4)
            sentences.append({
                "index": i, "number": i + 1, "text": s.text, "start": s.start, "end": s.end, "paragraph": s.para,
                "words": int(s.c("n_words")), "probability": p, "band": band, "level": _level(p, level_names),
                "confidence": conf_mod.sentence_confidence(float(win_words[i]), ag, bundle.get("length_table")),
                "signals": f, "explanation": summary_text(f, elevated=band == "ai") if band != "human" else None,
                "detectors": {k: round(v, 4) for k, v in sw.items()},
            })

        paragraphs = []
        for p_idx in sorted({s.para for s in doc.sentences}):
            m = np.array([s.para == p_idx for s in doc.sentences])
            paragraphs.append({"index": int(p_idx), "probability": float(np.sum(sp[m] * words[m]) / words[m].sum()),
                               "words": int(words[m].sum()), "sentences": [int(i) for i in np.flatnonzero(m)]})

        cp = consistency.detect_changepoints(doc.sentences, doc.K, window=cfg["changepoint"]["window"],
                                             min_segment=cfg["changepoint"]["min_segment"],
                                             permutations=cfg["changepoint"]["permutations"],
                                             alpha=cfg["changepoint"]["alpha"], extra=sp)
        for c in cp["changepoints"]:
            b = c["boundary_before_sentence"]
            c["ai_probability_before"] = float(np.mean(sp[max(0, b - 4):b]))
            c["ai_probability_after"] = float(np.mean(sp[b:b + 4]))

        detectors = []
        for name, v in det_doc.items():
            det = doc_ens.detectors.get(name) or doc_ens.text_detectors[name]
            detectors.append({
                "name": name, "label": DETECTOR_LABELS.get(name, name), "probability": v["prob"],
                "confidence": v["confidence"], "error_estimate": v["std"], "weight": doc_ens.weights.get(name, 0.0),
                "validation_brier": det.validation.get("oof_brier_balanced"),
                "validation_auc": det.validation.get("oof_roc_auc"), "n_features": len(getattr(det, "feature_names", [])),
            })
        detectors.sort(key=lambda d: -d["weight"])
        pvals = np.array([d["probability"] for d in detectors])
        disagreement = {
            "std": float(pvals.std()), "range": float(pvals.max() - pvals.min()),
            "most_ai": max(detectors, key=lambda d: d["probability"])["label"],
            "most_human": min(detectors, key=lambda d: d["probability"])["label"],
            "note": ("The detectors largely agree." if pvals.std() < 0.1 else
                     "The detectors disagree substantially; treat the ensemble estimate with extra caution."
                     if pvals.std() >= 0.2 else "The detectors partly disagree."),
        }

        level = ("strongly" if doc_prob >= 0.8 else "moderately" if doc_prob >= 0.6 else
                 "weakly" if doc_prob >= 0.4 else "not notably")
        result = {
            "ai_associated_share": share_ai, "human_associated_share": share_hu, "uncertain_share": share_un,
            "document_probability": doc_prob, "estimated_ai_share": est_share,
            "confidence": conf["value"], "confidence_level": conf["level"], "confidence_components": conf["components"],
            "evidence_level": evidence, "evidence_text": conf_mod.EVIDENCE_TEXT[evidence],
            "hybrid": hybrid, "interpretation": INTERPRETATION.format(level=level),
            "bands": {"t_high": t_hi, "t_low": t_lo},
            "share_definition": ("Shares are fractions of words in sentences whose calibrated sentence probability is "
                                 f">= {t_hi:.2f} (AI-associated), <= {t_lo:.2f} (human-associated) or in between "
                                 "(uncertain/mixed)."),
        }
        stats = self._statistics(dfeat, doc_ens, doc, fx)
        series = self._series(doc, sp, cp, fx)
        out = {**base, "status": "ok", "result": result, "detectors": detectors, "disagreement": disagreement,
               "ensemble_weights": doc_ens.weights, "sentences": sentences, "paragraphs": paragraphs,
               "statistics": stats, "series": series, "transitions": cp,
               "explanation": {"document": doc_factors, "summary": summary_text(doc_factors, elevated=doc_prob >= 0.5),
                               "caveat": SIGNAL_CAVEAT,
                               "family_contributions": doc_families,
                               "method": "Baseline-replacement attribution on the deployed ensemble: each measurement is "
                                         "replaced by its median in human-written training documents and the change in "
                                         "the ensemble score is its contribution (interactions are ignored)."},
               "model": {"version": bundle["model_version"], "feature_version": bundle["feature_version"],
                         "profile": bundle["spec"]["profile"], "predictability_backend": bundle["spec"]["predictability"],
                         "semantic_backend": bundle["spec"]["semantic"],
                         "evidence_thresholds_derived": bool(bundle.get("evidence_derived")),
                         "calibration": {"document": bundle["calibration_reports"]["document"]["selected"],
                                         "window": bundle["calibration_reports"]["window"]["selected"]}}}
        out["meta"].update(sentence_count=n, paragraph_count=doc.n_paragraphs, word_count=n_words)
        return out

    def _statistics(self, f: dict, ens, doc, fx) -> dict:
        pop = ens.population

        def item(key, label, fmt="{:.3f}", note=""):
            v = f.get(key, float("nan"))
            ref = pop.get(key, {})
            h, a = ref.get("human"), ref.get("ai")
            return {"key": key, "label": label, "value": None if v != v else float(v),
                    "display": "n/a" if v != v else fmt.format(v),
                    "human_percentile": None if (h is None or v != v) else round(percentile_in(h["quantiles"], v), 1),
                    "human_median": h["quantiles"][10] if h else None, "ai_median": a["quantiles"][10] if a else None,
                    "note": note}

        pred_unit = "bits/word (unigram proxy)" if fx.pred_kind == "unigram" else "bits/token"
        mean_bits = f.get("ppl_sent_mean", float("nan"))
        ppl = 2 ** mean_bits if mean_bits == mean_bits else float("nan")
        bur = pop.get("bur_ppl_sent_std", {}).get("human")
        burst_score = (percentile_in(bur["quantiles"], f["bur_ppl_sent_std"]) / 100
                       if bur and f.get("bur_ppl_sent_std") == f.get("bur_ppl_sent_std") else None)
        return {
            "perplexity": {"summary": {"perplexity": None if ppl != ppl else float(ppl), "unit": pred_unit,
                                       "backend": fx.pred_kind},
                           "items": [item("ppl_sent_mean", f"Mean sentence surprisal ({pred_unit})"),
                                     item("ppl_token_median", "Median word surprisal"),
                                     item("ppl_sent_min", "Lowest sentence surprisal")]},
            "burstiness": {"summary": {"normalized_score": burst_score,
                                       "definition": "percentile of sentence-to-sentence predictability variation "
                                                     "among human-written reference documents (0 = very uniform)"},
                           "items": [item("bur_ppl_sent_std", "Sentence perplexity std."),
                                     item("bur_ppl_sent_B", "Perplexity burstiness index B"),
                                     item("bur_sent_len_cv", "Sentence-length variation (CV)"),
                                     item("bur_sent_len_B", "Sentence-length burstiness B"),
                                     item("bur_syntax_cv", "Syntactic variation (CV of depth)")]},
            "stylometry": {"items": [item("sty_sent_len_mean", "Average sentence length (words)", "{:.1f}"),
                                     item("sty_sent_len_median", "Median sentence length", "{:.1f}"),
                                     item("sty_word_len_mean", "Average word length", "{:.2f}"),
                                     item("sty_function_frac", "Function-word share"),
                                     item("sty_contraction_rate", "Contractions per 100 words", "{:.2f}"),
                                     item("sty_pron1s_frac", "First-person singular share", "{:.4f}"),
                                     item("sty_pos_adv_frac", "Adverb share"),
                                     item("sty_pos_adj_frac", "Adjective share")]},
            "punctuation": {"items": [item("pun_comma_rate", "Commas per 100 words", "{:.2f}"),
                                      item("pun_semicolon_rate", "Semicolons per 100 words", "{:.2f}"),
                                      item("pun_colon_rate", "Colons per 100 words", "{:.2f}"),
                                      item("pun_dash_rate", "Dashes per 100 words", "{:.2f}"),
                                      item("pun_paren_rate", "Parentheses per 100 words", "{:.2f}"),
                                      item("pun_type_entropy", "Punctuation variety (entropy)")],
                            "note": "No punctuation mark is treated as evidence on its own."},
            "vocabulary": {"items": [item("voc_mattr", "Moving-average type-token ratio"),
                                     item("voc_mtld", "MTLD lexical diversity", "{:.1f}"),
                                     item("voc_hapax_ratio", "Hapax legomena ratio"),
                                     item("voc_zipf_mean", "Word commonness (Zipf)", "{:.2f}"),
                                     item("voc_rare_frac", "Rare-word share"),
                                     item("voc_heaps_slope", "Vocabulary growth (Heaps)")]},
            "syntax": {"items": [item("syn_depth_mean", "Dependency-tree depth", "{:.2f}"),
                                 item("syn_clauses_per_sent", "Clauses per sentence", "{:.2f}"),
                                 item("syn_subord_per_sent", "Subordinate clauses per sentence", "{:.2f}"),
                                 item("syn_passive_frac", "Passive-voice sentences"),
                                 item("syn_opening_entropy", "Sentence-opening variety"),
                                 item("syn_pos_trigram_diversity", "Syntactic diversity")]},
            "repetition": {"items": [item("rep_trigram", "Repeated 3-word phrases"),
                                     item("rep_pos_trigram", "Repeated grammatical templates"),
                                     item("rep_sentence_template", "Repeated sentence structures"),
                                     item("tra_per_sent", "Transition words per sentence"),
                                     item("tra_diversity", "Transition diversity"),
                                     item("tra_initial_frac", "Sentence-initial transitions")]},
            "semantic": {"items": [item("sem_adj_mean", "Neighbouring-sentence similarity"),
                                   item("sem_local_gain", "Local coherence gain"),
                                   item("sem_topic_jump_rate", "Topic-jump rate"),
                                   item("sem_redundancy", "Semantic redundancy"),
                                   item("sem_compression", "Semantic compression")],
                         "backend": fx.sem_kind},
            "consistency": {"items": [item("style_drift_mean", "Style drift (mean JS divergence)"),
                                      item("style_drift_max", "Largest style shift")]},
            "paragraphs": {"items": [item("par_len_cv", "Paragraph-length variation (CV)"),
                                     item("par_ppl_std", "Paragraph perplexity variation")],
                           "note": "Descriptive only: paragraph formatting is not used by the classifiers."},
        }

    def _series(self, doc, sp, cp, fx) -> dict:
        R = doc.sentences
        surpr = [float(np.mean(s.surprisal)) if len(s.surprisal) else None for s in R]
        sl = [int(s.c("n_words")) for s in R]
        roll_ppl, roll_len, roll_mattr = [], [], []
        for i in range(len(R)):
            idx = fx.window_indices(len(R), i)
            v = [surpr[j] for j in idx if surpr[j] is not None]
            roll_ppl.append(float(np.std(v)) if len(v) > 1 else None)
            lens = [sl[j] for j in idx]
            roll_len.append(float(np.std(lens) / max(np.mean(lens), 1e-9)) if len(lens) > 1 else None)
            roll_mattr.append(float(mattr([w for j in idx for w in R[j].words])))
        drift = consistency.style_drift(R, fx.style_window, fx.style_stride)
        return {"sentence_probability": [float(x) for x in sp], "sentence_surprisal": surpr,
                "rolling_perplexity_std": roll_ppl, "rolling_length_cv": roll_len,
                "rolling_mattr": roll_mattr, "sentence_lengths": sl,
                "style_drift": {"centers": drift["centers"], "values": drift["series"]},
                "changepoint_statistic": cp.get("series", []), "changepoint_threshold": cp.get("threshold")}


def _level(p: float, levels) -> str:
    for lo, hi, name in levels:
        if lo <= p < hi:
            return name
    return levels[-1][2]


@lru_cache(maxsize=1)
def default_analyzer() -> Analyzer:
    return Analyzer()


def main(argv=None) -> int:
    from .io_utils import read_document
    from .report import write_report

    ap = argparse.ArgumentParser(description="Analyse a document (TXT/DOCX/PDF) for AI-associated patterns.")
    ap.add_argument("path", help="file to analyse, or '-' to read text from stdin")
    ap.add_argument("--report", help="write a report (.html, .pdf or .json)")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None)
    args = ap.parse_args(argv)
    if args.path == "-":
        text, title = sys.stdin.read(), "stdin"
    else:
        text, title = read_document(args.path), Path(args.path).name
    res = Analyzer(device=args.device).analyze(text, title=title)
    if res["status"] != "ok":
        print(f"[{res['status']}] {res.get('message', '')}")
        return 2
    r = res["result"]
    print(f"Document: {title}   words: {res['meta']['word_count']}   language: {res['meta']['language']}")
    print(f"AI-associated: {r['ai_associated_share']:.0%}   Human-associated: {r['human_associated_share']:.0%}   "
          f"Uncertain/mixed: {r['uncertain_share']:.0%}")
    print(f"Document-level AI probability: {r['document_probability']:.0%}   Confidence: {r['confidence']:.0%} "
          f"({r['confidence_level']})   Evidence: {r['evidence_text']}")
    print(f"Category: {r['hybrid']['label']}   - {r['interpretation']}")
    for d in res["detectors"]:
        print(f"  {d['label']:<55s} {d['probability']:6.1%}  (weight {d['weight']:.2f})")
    print(DISCLAIMER)
    if args.report:
        write_report(res, args.report)
        print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
