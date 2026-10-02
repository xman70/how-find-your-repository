# Model card - `en-20261002-783106e4aa39-nonnative`

**Task.** Estimate whether English text shows statistical patterns associated with AI-generated writing (document and sentence level). Assists human review; not an authorship determination.

**Training data.** M4 + Ghostbuster (+ half of ETS/PELIC/Lang-8 non-native human essays), dataset version `783106e4aa39`; 16920 document rows, 60000 sentence windows. Generators: bloomz, chatgpt, claude, cohere, davinci, dolly, flant5, llama (all 2023-era). Domains: arxiv, essay, ets, lang8, peerread, pelic, reddit, reuter, wikihow, wikipedia, wp.

**Architecture.** Lite feature profile (unigram-surprisal perplexity proxy, lexical semantic vectors, spaCy syntax); 9 detectors; logistic meta-classifier with learned weights statistical 0.65, neural 0.35; calibration: isotonic (document), isotonic (sentence).

## Measured performance (held-out test split, threshold 0.5)

ROC-AUC 0.972 · accuracy 0.907 · FPR 9.2% · FNR 9.4% · ECE 0.012 · TPR at 1% FPR 0.717.

Unseen generator families (leave-one-out ROC-AUC): anthropic 0.846, bigscience 0.563, cohere 0.953, databricks 0.876, google 0.991, meta 1.000, openai 0.947.

Unseen domains (leave-one-out ROC-AUC): arxiv 0.856, essay 0.955, peerread 0.966, reddit 0.961, reuter 0.969, wikihow 0.871, wikipedia 0.841, wp 0.837.

Highest false-positive rates in the audit: liang_CS224N_real_145 44%, gb_pelic 19%, test_wikipedia_highly_edited 16%, test_arxiv_scientific 16%.

Evidence thresholds (words): {"low": 75, "moderate": 150, "reliable": 300}.

## Intended use and limits

* Use as one input to a human review, never as sole evidence.
* Not validated for languages other than English (Greek is refused until a Greek model is trained).
* Not validated on text from generators released after 2023, on dyslexic writers, or on translated text.
* Elevated false-positive rates for formulaic scientific/legal writing and for short texts (see report).

Full results: `reports/evaluation/en/EVALUATION_REPORT.md`.
