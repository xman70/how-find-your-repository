"""Render reports/evaluation/<lang>/evaluation_results.json into EVALUATION_REPORT.md + figures.

Every number in the report is read from the JSON produced by evaluate.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidetect.visualization import grouped_bars, length_figure, reliability_figure  # noqa: E402


def f3(v):
    return "n/a" if v is None else f"{v:.3f}"


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def ci(c):
    return "" if not c else f" [{100 * c[0]:.1f}–{100 * c[1]:.1f}]"


def ci3(c):
    return "" if not c else f" [{c[0]:.3f}–{c[1]:.3f}]"


def main(lang: str = "en", fp_compare: str | None = None) -> None:
    d = ROOT / "reports" / "evaluation" / lang
    R = json.loads((d / "evaluation_results.json").read_text())
    figs = d / "figures"
    figs.mkdir(exist_ok=True)
    L = []
    w = L.append
    t = R["test"]["overall"]
    tr = R["training"]
    w(f"# Evaluation report ({lang})\n")
    w(f"Model `{R['model_version']}` · evaluated {R['date']} · dataset version `{tr['dataset_version']}` · "
      f"seed {tr['seed']}. All numbers below are produced by `evaluate.py` and read from "
      f"`evaluation_results.json`; nothing is hand-edited.\n")
    w("> **Scope.** These results describe *this* model on *these* datasets. All AI text in training and test "
      "comes from 2023-era generators (GPT-3.5, text-davinci-003, Cohere, Dolly-v2, BLOOMZ, Flan-T5, LLaMA, "
      "Claude 2023). Performance on newer models, other domains, other languages and real classroom populations "
      "is not established by this report. Probabilities are calibrated for a 50 % prior.\n")
    w("## 1. Held-out test set (document level)\n")
    w(f"Test documents: {t['n']} ({t['n_neg']} human, {t['n_pos']} AI), grouped so that no prompt or source text "
      f"appears in both training and test. 95 % CIs from a group-level bootstrap.\n")
    w("| Metric | Value | 95 % CI |\n|---|---|---|")
    for k, label in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall (TPR)"), ("f1", "F1"),
                     ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("fpr", "False-positive rate"),
                     ("fnr", "False-negative rate")):
        w(f"| {label} | {f3(t.get(k))} | {ci3(t['ci95'].get(k)) if t.get('ci95') else ''} |")
    w(f"| Brier score (balanced) | {f3(t['brier_balanced'])} | |\n| Expected calibration error (balanced) | {f3(t['ece_balanced'])} | |")
    w(f"| TPR at 1 % FPR | {f3(t.get('tpr_at_fpr_1pct'))} | |\n| TPR at 5 % FPR | {f3(t.get('tpr_at_fpr_5pct'))} | |\n")
    w("Threshold 0.5 on the calibrated probability.\n")
    w("### By domain\n\n| Domain | n | ROC-AUC | FPR (human flagged) | FNR (AI missed) |\n|---|---|---|---|---|")
    for dom, m in R["test"]["by_domain"].items():
        w(f"| {dom} | {m['n']} | {f3(m['roc_auc'])} | {pct(m['fpr'])}{ci(m.get('fpr_ci95'))} | {pct(m['fnr'])} |")
    w("\n### AI detection rate by generator (test split, in-distribution)\n\n| Generator | n | Detected | Mean probability |\n|---|---|---|---|")
    for g, m in R["test"]["ai_detection_by_generator"].items():
        w(f"| {g} | {m['n']} | {pct(m['detection_rate'])}{ci(m['detection_rate_ci95'])} | {f3(m['mean_probability'])} |")
    h = R["test"]["hybrids_document_level"]
    w(f"\nSpliced human/AI hybrids (target: AI share >= 50 %): n={h['n']}, ROC-AUC {f3(h['roc_auc'])}, "
      f"accuracy {f3(h['accuracy'])}.\n")

    if "cross_generator" in R:
        w("## 2. Cross-generator generalisation (leave-one-family-out)\n")
        w("The full pipeline (detectors, meta-classifier, calibration) is retrained without any text from the held-out "
          "family and tested on that family's AI texts plus the held-out human texts. *In-distribution* is the shipped "
          "model (which saw the family) on the same test documents.\n")
        w("| Held-out family | Test n | ROC-AUC held-out | 95 % CI | TPR held-out | FPR held-out | ROC-AUC in-distribution |\n|---|---|---|---|---|---|---|")
        cats, s_out, s_in = [], [], []
        for fam, r in R["cross_generator"].items():
            ho, ind = r["held_out"], r["in_distribution_reference"]
            w(f"| {fam} | {r['n_test']} | {f3(ho['roc_auc'])} | {ci3(ho['ci95'].get('roc_auc'))} | {pct(ho['recall'])} | "
              f"{pct(ho['fpr'])} | {f3(ind['roc_auc'])} |")
            cats.append(fam)
            s_out.append(ho["roc_auc"])
            s_in.append(ind["roc_auc"])
        (figs / "cross_generator.png").write_bytes(grouped_bars(
            cats, {"held out (unseen family)": s_out, "in-distribution": s_in}, "ROC-AUC by generator family", "ROC-AUC",
            ylim=(0.5, 1.0)))
        w("\n![cross-generator](figures/cross_generator.png)\n")
    if "cross_domain" in R:
        w("## 3. Cross-domain generalisation (leave-one-domain-out)\n")
        w("| Held-out domain | Test n | ROC-AUC held-out | FPR held-out | FNR held-out | ROC-AUC in-distribution |\n|---|---|---|---|---|---|")
        cats, s_out, s_in = [], [], []
        for dom, r in R["cross_domain"].items():
            ho, ind = r["held_out"], r["in_distribution_reference"]
            w(f"| {dom} | {r['n_test']} | {f3(ho['roc_auc'])}{ci3(ho['ci95'].get('roc_auc'))} | {pct(ho['fpr'])} | "
              f"{pct(ho['fnr'])} | {f3(ind['roc_auc'])} |")
            cats.append(dom)
            s_out.append(ho["roc_auc"])
            s_in.append(ind["roc_auc"])
        (figs / "cross_domain.png").write_bytes(grouped_bars(
            cats, {"held out (unseen domain)": s_out, "in-distribution": s_in}, "ROC-AUC by domain", "ROC-AUC",
            ylim=(0.3, 1.0)))
        w("\n![cross-domain](figures/cross_domain.png)\n")

    c = R["calibration"]
    w("## 4. Calibration\n")
    w("Calibrators are fitted on the calibration split (class-balanced) and evaluated once on the test split.\n")
    w("| Level | Method | Brier | ECE | selected on calibration split |\n|---|---|---|---|---|")
    for level in ("document", "window"):
        for m in ("none", "platt", "isotonic"):
            e = c[level]["test"][m]
            w(f"| {level} | {m} | {f3(e['brier'])} | {f3(e['ece'])} | {'yes' if c[level]['selected_on_calibration_split'] == m else ''} |")
        (figs / f"reliability_{level}.png").write_bytes(reliability_figure(
            {m: c[level]["test"][m]["reliability"] for m in ("none", "platt", "isotonic")},
            f"Reliability ({level} level, test split)"))
    w("\n![reliability document](figures/reliability_document.png) ![reliability window](figures/reliability_window.png)\n")
    w("Reliability bins (document level, selected method):\n\n| Bin | n | mean predicted | observed AI share |\n|---|---|---|---|")
    sel = c["document"]["selected_on_calibration_split"]
    for b in c["document"]["test"][sel]["reliability"]["bins"]:
        if b["count"]:
            w(f"| {b['lo']:.1f}–{b['hi']:.1f} | {b['count']} | {f3(b['mean_pred'])} | {f3(b['frac_pos'])} |")

    if "end_to_end" in R:
        e = R["end_to_end"]
        s = e["sentence_level"]
        w("\n## 5. Sentence level, AI share, hybrid classes, transitions (end-to-end inference)\n")
        w(f"Run through the same `Analyzer` the application uses, on {e['n_documents']} test documents "
          f"({e['n_sentences']} sentences: spliced hybrids + pure documents).\n")
        w(f"* Sentence-level ROC-AUC **{f3(s['roc_auc'])}**, accuracy {f3(s['accuracy'])}, FPR {pct(s['fpr'])}, "
          f"FNR {pct(s['fnr'])}, ECE (balanced) {f3(s['ece_balanced'])}.")
        a = e["ai_share"]
        w(f"* AI-share estimation: mean absolute error {pct(a['mae_estimated_share'])} (calibrated estimate), "
          f"{pct(a['mae_band_share'])} (share of words in the AI-associated band); correlation {f3(a['correlation_estimated'])}.")
        hc = e["hybrid_classes"]
        w(f"* Five-way category: accuracy when a category is assigned {f3(hc['accuracy_when_decided'])}, "
          f"within one category {f3(hc['within_one_class_when_decided'])}, 'Unknown' rate {pct(hc['unknown_rate'])}.")
        cp = e["changepoints"]
        w(f"* Change points (±2 sentences): precision {f3(cp['precision_within_2_sentences'])}, recall "
          f"{f3(cp['recall_within_2_sentences'])} over {cp['true_boundaries']} true boundaries; false-alarm rate on "
          f"pure documents {pct(cp['false_alarm_rate_pure_documents'])} (n={cp['n_pure']}).\n")
        w("Confusion matrix (rows = true category by actual AI share, columns = predicted):\n")
        w("| true \\ predicted | " + " | ".join(hc["labels_pred"]) + " |\n|---|" + "---|" * len(hc["labels_pred"]))
        for lab, row in zip(hc["labels_true"], hc["confusion"]):
            w(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
        w("\nConfidence validity (document decision correct vs confidence level):\n\n| Confidence | n | accuracy |\n|---|---|---|")
        for k, v in e["confidence_validity"].items():
            w(f"| {k} | {v['n']} | {f3(v['accuracy'])} |")

    if "length" in R:
        Ln = R["length"]
        w("\n## 6. Minimum text length (experimentally derived thresholds)\n")
        w("Test documents truncated at sentence boundaries; thresholds are the smallest lengths from which ROC-AUC stays "
          f"within the criteria {json.dumps(Ln['criteria'])} (FPR at the 0.5 threshold).\n")
        w("| Words | n | ROC-AUC | FPR | FNR |\n|---|---|---|---|---|")
        for r in Ln["rows"]:
            w(f"| {r['words']} | {r['n']} | {f3(r['roc_auc'])} | {pct(r['fpr'])} | {pct(r['fnr'])} |")
        th = Ln["derived_thresholds"]
        w(f"\nDerived: *Insufficient evidence* below {th['low']} words · *Low confidence* {th['low']}–{th['moderate']} · "
          f"*Moderate* {th['moderate']}–{th['reliable']} · *More reliable* from {th['reliable']} words "
          f"(`None` = criterion never met in the tested range). Stored in the model bundle.\n")
        if Ln.get("initial_derivation"):
            ini = Ln["initial_derivation"]
            w(f"Note: the first derivation used AUC-only criteria {json.dumps(ini['criteria'])}, giving "
              f"{json.dumps(ini['derived_thresholds'])}. Because that labelled texts with a measured false-positive rate "
              "above 15 % as 'more reliable', false-positive limits were added to the criteria after the study; the "
              "thresholds above use the revised criteria.\n")
        (figs / "length.png").write_bytes(length_figure(Ln["rows"], th))
        w("![length](figures/length.png)\n")

    w("## 7. Baseline model comparison (same features and splits)\n")
    w("| Model | ROC-AUC | Accuracy | F1 | FPR | FNR | ECE |\n|---|---|---|---|---|---|---|")
    from aidetect.models import BASELINE_LABELS

    for k, m in R["baselines"].items():
        if "skipped" in m:
            w(f"| {BASELINE_LABELS.get(k, k)} | skipped ({m['skipped']}) | | | | | |")
            continue
        w(f"| {BASELINE_LABELS.get(k, k)} | {f3(m['roc_auc'])} | {f3(m['accuracy'])} | {f3(m['f1'])} | {pct(m['fpr'])} | "
          f"{pct(m['fnr'])} | {f3(m['ece_balanced'])} |")

    w("\n## 8. Adversarial and robustness categories\n")
    w("Detection rate = share of AI documents with calibrated probability >= 0.5; flag rate = share of human documents "
      "above 0.5. *Reference* = the same documents before perturbation (when they are in the test split). "
      "`sim_*` rows are local simulations, `perturb_*` rows are Ghostbuster perturbations (10 edits per document).\n")
    w("| Category | n (AI/human) | AI detection rate | Reference | Human flag rate | Reference | ROC-AUC |\n|---|---|---|---|---|---|---|")
    for k, a in R["adversarial"].items():
        w(f"| {k} | {a['n_ai']}/{a['n_human']} | {pct(a.get('ai_detection_rate'))}{ci(a.get('ai_detection_rate_ci95'))} | "
          f"{pct(a.get('reference_unperturbed_ai_detection_rate'))} | {pct(a.get('human_flag_rate'))}{ci(a.get('human_flag_rate_ci95'))} | "
          f"{pct(a.get('reference_unperturbed_human_flag_rate'))} | {f3(a.get('roc_auc'))} |")

    fp = R["fp_audit"]
    w("\n## 9. Critical false-positive audit\n")
    w(f"Human-written documents only. In-distribution reference FPR (all human test documents): "
      f"**{pct(fp['reference_in_distribution_fpr'])}**. Rows marked *unseen* come from populations never used in "
      "training. The ratio compares each group's FPR with the reference.\n")
    w("| Group | Writers | Genre | n | FPR | 95 % CI | Ratio vs reference | Mean probability | Mean AI-associated share |\n|---|---|---|---|---|---|---|---|---|")
    cats, vals, errs = [], [], []
    for k, g in fp["groups"].items():
        tag = "" if g.get("in_distribution") else " (unseen)"
        w(f"| {k}{tag} | {g['writer_group']} | {g['genre']} | {g['n']} | {pct(g['fpr'])} | {ci(g['fpr_ci95'])} | "
          f"{f3(g.get('fpr_ratio_vs_in_distribution'))} | {f3(g['mean_probability'])} | {pct(g.get('mean_ai_associated_share'))} |")
        cats.append(k.replace("liang_", "").replace("gb_", "").replace("_real", "")[:22])
        vals.append(g["fpr"])
        errs.append(g["fpr_ci95"] or [g["fpr"], g["fpr"]])
    (figs / "fp_audit.png").write_bytes(grouped_bars(cats, {"FPR": vals}, "False-positive rate by human population",
                                                     "FPR", ylim=(0, max(0.2, max(vals) * 1.3)),
                                                     ref=fp["reference_in_distribution_fpr"], ref_label="in-distribution",
                                                     errors={"FPR": errs}))
    w("\n![fp audit](figures/fp_audit.png)\n")
    w("Not tested: " + "; ".join(fp["not_tested"]) + ".\n")
    if fp_compare:
        w(fp_compare)

    det = R["detectors"]
    w("## 10. Detector disagreement and feature importance\n")
    w("| Detector | Test ROC-AUC | Mean committee std | Learned weight |\n|---|---|---|---|")
    for k, v in det["per_detector"].items():
        w(f"| {k} | {f3(v['test_roc_auc'])} | {f3(v['mean_committee_std'])} | {f3(v['weight'])} |")
    w(f"\nEnsemble accuracy when detectors agree (spread below median): {f3(det['accuracy_when_detectors_agree'])}; "
      f"when they disagree: {f3(det['accuracy_when_detectors_disagree'])}.\n")
    w("Top features of the gradient-boosting detector (permutation importance, ROC-AUC drop):\n")
    w("| Feature | AUC drop |\n|---|---|")
    for r in det["statistical_model_permutation_importance_top20"]:
        w(f"| {r['feature']} | {r['auc_drop']:.4f} |")
    sa = R["shortcut_audit"]
    w(f"\n## 11. Shortcut (artefact) audit\n\n{sa['n_flagged']} (domain, feature) pairs reach single-feature ROC-AUC >= "
      f"{sa['threshold']} in the training split.\n")
    if sa["flagged"]:
        w("| Domain | Feature | AUC | Direction |\n|---|---|---|---|")
        for r in sa["flagged"][:20]:
            w(f"| {r['domain']} | {r['feature']} | {f3(r['single_feature_auc'])} | {r['direction']} |")
    w("\n## Training summary\n")
    w(f"Document rows {tr['n_doc_train']}, window rows {tr['n_window_train']}; generators in training: "
      f"{', '.join(tr['generators_in_training'])}; domains: {', '.join(tr['domains_in_training'])}. Learned "
      f"document-level ensemble weights: " + ", ".join(f"{k} {v:.2f}" for k, v in tr["doc_weights"].items() if v > 0) +
      " (all other detectors received weight 0 from the meta-classifier: redundant given these).")
    w(f"\nEvaluation runtime: {R.get('seconds')} s.")
    (d / "EVALUATION_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {d / 'EVALUATION_REPORT.md'}")
    write_model_card(lang, R)


def write_model_card(lang: str, R: dict) -> None:
    t, tr = R["test"]["overall"], R["training"]
    th = R.get("evidence_thresholds", {})
    cg = R.get("cross_generator", {})
    cd = R.get("cross_domain", {})
    fp = R["fp_audit"]["groups"]
    worst_fp = sorted(((k, g["fpr"]) for k, g in fp.items()), key=lambda kv: -kv[1])[:4]
    lines = [
        f"# Model card - `{R['model_version']}`\n",
        "**Task.** Estimate whether English text shows statistical patterns associated with AI-generated writing "
        "(document and sentence level). Assists human review; not an authorship determination.\n",
        f"**Training data.** M4 + Ghostbuster (+ half of ETS/PELIC/Lang-8 non-native human essays), dataset version "
        f"`{tr['dataset_version']}`; {tr['n_doc_train']} document rows, {tr['n_window_train']} sentence windows. "
        f"Generators: {', '.join(tr['generators_in_training'])} (all 2023-era). Domains: {', '.join(tr['domains_in_training'])}.\n",
        "**Architecture.** Lite feature profile (unigram-surprisal perplexity proxy, lexical semantic vectors, spaCy "
        "syntax); 9 detectors; logistic meta-classifier with learned weights "
        + ", ".join(f"{k} {v:.2f}" for k, v in tr["doc_weights"].items() if v > 0)
        + f"; calibration: {tr['doc_calibration']['selected']} (document), {tr['window_calibration']['selected']} (sentence).\n",
        "## Measured performance (held-out test split, threshold 0.5)\n",
        f"ROC-AUC {t['roc_auc']:.3f} · accuracy {t['accuracy']:.3f} · FPR {100*t['fpr']:.1f}% · FNR {100*t['fnr']:.1f}% · "
        f"ECE {t['ece_balanced']:.3f} · TPR at 1% FPR {t.get('tpr_at_fpr_1pct', float('nan')):.3f}.\n",
    ]
    if cg:
        aucs = {k: v["held_out"]["roc_auc"] for k, v in cg.items()}
        lines.append("Unseen generator families (leave-one-out ROC-AUC): " + ", ".join(f"{k} {v:.3f}" for k, v in aucs.items()) + ".\n")
    if cd:
        aucs = {k: v["held_out"]["roc_auc"] for k, v in cd.items()}
        lines.append("Unseen domains (leave-one-out ROC-AUC): " + ", ".join(f"{k} {v:.3f}" for k, v in aucs.items()) + ".\n")
    lines += [
        "Highest false-positive rates in the audit: " + ", ".join(f"{k} {100*v:.0f}%" for k, v in worst_fp) + ".\n",
        f"Evidence thresholds (words): {json.dumps(th)}.\n",
        "## Intended use and limits\n",
        "* Use as one input to a human review, never as sole evidence.",
        "* Not validated for languages other than English (Greek is refused until a Greek model is trained).",
        "* Not validated on text from generators released after 2023, on dyslexic writers, or on translated text.",
        "* Elevated false-positive rates for formulaic scientific/legal writing and for short texts (see report).",
        "\nFull results: `reports/evaluation/en/EVALUATION_REPORT.md`.",
    ]
    (ROOT / "models" / lang / "MODEL_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote models/{lang}/MODEL_CARD.md")


if __name__ == "__main__":
    extra = None
    if len(sys.argv) > 2:
        extra = Path(sys.argv[2]).read_text(encoding="utf-8")
    main(sys.argv[1] if len(sys.argv) > 1 else "en", extra)
