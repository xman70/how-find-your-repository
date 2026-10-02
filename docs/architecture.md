# Architecture and research design

## Principles

1. **Ensemble of independent signals**, never a single metric.
2. **Generalisation over in-distribution accuracy**: grouped splits, leave-one-generator-out and
   leave-one-domain-out retraining, adversarial and population-specific tests.
3. **Calibrated probabilities** on held-out data, with **confidence reported separately**.
4. **False-positive control**: thresholds chosen for a target human false-flag rate; audit of
   non-native, student, academic and legal writing.
5. **Explainability and honesty**: per-detector outputs, attributions, limitations in every report.
6. **Local-first**: everything runs on the user's machine; no network needed after installation.

## Pipeline

```
                INPUT DOCUMENT (paste / TXT / DOCX / PDF)
                                 |
                       TEXT PREPROCESSOR
   normalisation (artefact removal) -> language id -> paragraphs -> spaCy parse
                -> per-sentence records (counts, POS, depth, transitions, ...)
                                 |
          +-------------+--------+--------+--------------+
          |             |                 |              |
   Linguistic      Statistical        Semantic       Consistency
   stylometry,     predictability     sentence       style drift (JS),
   punctuation,    (perplexity),      similarity     change points
   syntax,         burstiness,        (coherence,    (permutation test)
   transitions     vocabulary,        redundancy,
                   repetition         compression)
          |             |                 |              |
          +-------------+--------+--------+--------------+
                                 |
             ML DETECTORS (each: score, confidence, features, error estimate)
   perplexity | burstiness | stylometric | vocabulary | syntax | semantic | repetition
   (logistic regression, one feature family each)
   statistical (gradient boosting, all features) | neural (MLP, all features)
   [optional] transformer (fine-tuned encoder on raw text)
                                 |
              META-CLASSIFIER (logistic regression on detector logits,
              weights learned from out-of-fold predictions, non-negative)
                                 |
              CALIBRATION LAYER (Platt vs isotonic, chosen by CV on the
              held-out calibration split, class-balanced = 50 % prior)
                                 |
     document probability    sentence probabilities (5-sentence windows)
                                 |
   bands (data-derived) -> AI / uncertain / human shares -> 5-way hybrid class
   confidence model (length reliability x agreement x coverage x stability)
                                 |
                 EXPLANATION ENGINE (baseline-replacement attributions)
                                 |
        VISUAL REPORT (web UI, heatmap, 8 graphs, HTML / PDF / JSON)

   SIMILARITY MODULE (separate): winnowing fingerprints, n-gram containment,
   TF-IDF cosine, sentence similarity against a local reference corpus
```

## Two levels, one feature code path

Each document is parsed once into sentence records. Any span - a 5-sentence window, a paragraph,
the whole document - is featurised by aggregating those records. The **document model** is trained
on whole documents (plus length-truncated copies); the **window model** is trained on centred
5-sentence windows with the target "is the centre sentence AI-written?", using spliced hybrids for
genuinely mixed contexts. Window-level calibration makes each sentence probability calibrated on
held-out data.

## Feature families (121 document-level model features)

| Family | Examples | Notes |
|---|---|---|
| Perplexity (predictability) | token / sentence surprisal mean, median, quantiles | lite profile: unigram surprisal (wordfreq); full profile: causal LM |
| Burstiness | sentence-to-sentence surprisal std / CV / range / B-index, token-level std, rolling std, sentence-length std / CV / B, vocabulary and syntactic variation | Goh-Barabasi B = (σ-μ)/(σ+μ) |
| Stylometry + punctuation | sentence/word length distribution, function words, pronouns by person, articles, conjunctions, POS rates, contractions, questions; comma/semicolon/colon/dash/parenthesis/quote/ellipsis rates, punctuation entropy and transition entropy | no punctuation mark is evidence on its own |
| Vocabulary | MATTR, MTLD, Yule's K, hapax/dis legomena on fixed chunks, Zipf sophistication, rare/OOV share, local repetition, Heaps growth exponent | length-normalised |
| Syntax | dependency depth, dependency distance, clauses, subordination, coordination, passive voice, fragments, opening-type entropy, POS-bigram entropy, POS-trigram diversity | spaCy parser |
| Repetition + transitions | n-gram repetition, repeated 5-grams, POS-template repetition, repeated sentence templates/openings, compression ratio; transition frequency, position, diversity, category entropy, comma use, spacing regularity | transitions are never treated as evidence by mere presence |
| Semantic | adjacent similarity mean/std/min, centroid focus, redundancy, local coherence gain, topic-jump rate, semantic compression (top eigenvalue share) | lite: hashed lexical vectors; full: sentence-transformer |
| Consistency (document only) | style drift mean/max/std (Jensen-Shannon over POS, punctuation, function-word, length distributions of sliding windows), change-point statistic | |
| Paragraph (reported only) | paragraph length CV, paragraph perplexity std, paragraph similarity | excluded from models: formatting artefacts |

## Why perplexity is not used alone

Low perplexity means "easy for a language model to predict". Decoding strategies make generated text
low-perplexity, but polished human prose, formulaic genres (legal, scientific, news), simple
vocabulary and memorised text are low-perplexity too; paraphrasing or high-temperature sampling
raises the perplexity of AI text. In the shipped model the perplexity-only detector reaches an
out-of-fold ROC-AUC of 0.67 at document level - far weaker than the multi-signal ensemble.

## Confidence vs likelihood

`confidence = length_reliability x (0.6 + 0.4 agreement) x (0.5 + 0.5 coverage^3) x (0.7 + 0.3 stability)`

* **length reliability** - interpolated from the measured ROC-AUC on truncated held-out documents;
* **agreement** - weighted spread of the detector probabilities;
* **coverage** - share of features inside the training population's 0.5-99.5 % range;
* **stability** - spread of each detector's cross-validation committee.

The evaluation checks whether accuracy increases with confidence level. **In the current release it does not** (High 0.76 vs Medium 0.90 on 891 end-to-end test documents; no Low cases), so the confidence value is shown with its components but must not be read as a validated reliability estimate. Recalibrating it against observed accuracy is the main open item.

## Hybrid classification

The AI share is the word-weighted mean of calibrated sentence probabilities, mapped through an
isotonic share calibrator fitted on held-out spliced documents with known share. Categories are
operational definitions on that share (config: 15 % / 50 % / 85 %); "Unknown" is returned when
confidence < 0.35 or the text is shorter than the measured "insufficient evidence" length.

## Model zoo / baselines

`evaluate.py` compares Logistic Regression, Random Forest, HistGradientBoosting, LightGBM, RBF-SVM and
an MLP on the same features and splits, against the stacked ensemble.

## Reproducibility

Seeds are fixed (config `training.seed`), the dataset version is a hash of the sample ids, the
feature version is a code constant checked when loading a model, and every training/evaluation run
is logged as JSON under `experiments/runs/`.
