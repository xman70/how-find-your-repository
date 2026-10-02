# AI Text Analysis

A local-first, research-grade system that **estimates** whether a document contains statistical patterns
associated with AI-generated writing. It combines ten families of linguistic and statistical signals in a
calibrated ensemble, reports sentence-, paragraph- and document-level probabilities with a separate
confidence estimate, shows where individual detectors disagree, and explains which signals drove the result.

> **AI-text detection is probabilistic. The results should not be used as sole evidence for academic
> misconduct, disciplinary action, or authorship determination.** The detector is designed to assist human
> review, not to replace it. This project is independently implemented; it is not affiliated with, certified
> by, or claimed to be as accurate as any commercial academic-integrity product.

<!-- RESULTS-SUMMARY -->

## What you get

For every analysed document:

1. AI-associated share, 2. human-associated share, 3. uncertain/mixed share (fractions of words, using
   band thresholds derived from held-out data)
4. sentence-level calibrated AI probability (5-sentence context window)
5. paragraph-level probability, 6. document-level calibrated probability
7. highlighted text and 8. a heatmap (blue = human-associated, gray = uncertain, red = AI-associated;
   click a sentence to see its signals)
9. confidence (separate from likelihood) with its components, and an evidence level based on text length
10. plain-language explanation of contributing signals ("statistical indicators, not proof")
11. statistics: perplexity, burstiness, stylometry, punctuation, vocabulary, syntax, repetition,
    semantics, style consistency - each with its percentile among human-written reference texts
12. model disagreement: every detector's probability, error estimate and learned ensemble weight
13. limitations and uncertainty warnings
14. exportable **PDF / HTML / JSON** reports

Plus a five-way category (Human / Human-assisted / AI-assisted / AI-generated / Unknown), potential
authorship/style transitions (permutation-tested change points), and a **separate** similarity /
plagiarism module that compares text with a local reference corpus.

## Quick start (Windows)

1. Install **Python 3.12** (3.10-3.13 work) from <https://www.python.org/downloads/windows/> and tick
   *Add python.exe to PATH*.
2. Download or clone this repository.
3. Double-click **`START.bat`**. The first run creates `.venv`, installs dependencies (a few minutes),
   verifies the installation and opens <http://127.0.0.1:8765/> in your browser.
4. Paste text or upload a DOCX / PDF / TXT file, click **Analyze**, explore the heatmap and tabs,
   export a report.

`START.bat full` additionally installs the optional neural components (PyTorch, transformers).
Linux / macOS: `./start.sh`.

No GPU is needed. Nothing is uploaded: **Privacy mode: Local processing** (the server listens on
127.0.0.1 only; analysis history stores results and a text hash, not the text, by default).

## Command line

```bash
python predict.py essay.docx --report essay_report.pdf   # analyse a file, write PDF/HTML/JSON report
python app.py                                            # start the web app
python -m aidetect.verify                                 # check the installation
```

## Reproducing the model

```bash
python scripts/fetch_public_datasets.py          # M4, Ghostbuster, Liang et al. -> data/external/ (not committed)
python -m aidetect.build_dataset --lang en --jobs 4   # splits, hybrids, adversarial + audit sets, features
python train.py --lang en                        # trains models/en/bundle.joblib (~11 min on 4 CPU cores)
python evaluate.py --lang en                     # all studies; derives length thresholds into the bundle
python scripts/make_evaluation_report.py en      # renders reports/evaluation/en/EVALUATION_REPORT.md
```

Every training and evaluation run is logged to `experiments/runs/*.json` (dataset version, feature
version, model version, hyperparameters, seed, validation and test results).

## How it works

```
document -> preprocessing (artefact removal, language id, spaCy parse, sentence records)
         -> features: perplexity | burstiness | stylometry+punctuation | vocabulary | syntax
                      | repetition+transitions | semantics | style drift + change points
         -> detectors (each: score, confidence, features, error estimate)
              7 single-family logistic models, gradient boosting (all features), MLP (all features),
              optional fine-tuned transformer
         -> meta-classifier (weights learned from out-of-fold predictions)
         -> calibration (Platt vs isotonic, chosen on held-out data)
         -> bands, shares, hybrid category, confidence, explanations, report
```

Details: [`docs/architecture.md`](docs/architecture.md). Dataset design, sources, leakage controls and
known gaps: [`data/README.md`](data/README.md). Full measured results:
[`reports/evaluation/en/EVALUATION_REPORT.md`](reports/evaluation/en/EVALUATION_REPORT.md).

### Feature profiles

| Profile | Perplexity backend | Semantic backend | Status |
|---|---|---|---|
| **lite** (shipped) | unigram surprisal from corpus word frequencies (`wordfreq`) | hashed lexical vectors | trained and evaluated |
| full | causal language model (GPT-2 by default) | multilingual sentence-transformer | implemented; needs `requirements-optional.txt` and model downloads; **not trained or evaluated here** |

The build environment for this release could not download neural model weights (Hugging Face was
unreachable), so the shipped model uses the lite profile: its "perplexity" is a **proxy** (how common the
words are), not language-model perplexity. A model trained with one profile refuses to run with the other.

## Greek support

The preprocessing and feature pipeline supports Greek (spaCy `el_core_news_sm`, Greek function words,
transition markers, accent-insensitive matching, Greek word frequencies). **No Greek model has been trained
or validated**, so the application detects Greek text and declines to score it instead of applying the
English model, whose performance does not transfer across languages. To build one:

1. collect Greek human texts with permission (academic, student, informal, translated) and Greek AI texts,
   e.g. `python -m aidetect.generation --backend ollama --model <local model> --tasks write_el,translate_el ...`;
2. `python -m aidetect.build_dataset --lang el --local greek_corpus.jsonl`;
3. `python train.py --lang el` and `python evaluate.py --lang el` - English and Greek results are reported
   separately.

## Extending the training data (multi-model generation)

`python -m aidetect.generation` creates topic-matched AI counterparts and borderline categories
(continuation, paraphrase of human or AI text, AI polishing, translation, back-translation, Greek writing)
with several model families: Claude (official `anthropic` SDK; refusal fallback enabled, and the model that
actually served each response is recorded), any OpenAI-compatible endpoint (OpenAI, Mistral, vLLM, LM
Studio), Google Gemini, and local models through Ollama (Llama, Mistral, Qwen ... - fully local). Cloud
backends only run with `--allow-external-upload`. Retrain and re-run the cross-generator study whenever new
generator families are added; evaluate on families that were **not** used for training.

## Project layout

```
START.bat / start.sh       launchers            config/config.yaml   all settings
app.py train.py evaluate.py predict.py          entry points
src/aidetect/              package: preprocessing, features (+ signals/), models, calibration, confidence,
                           hybrid, explain, predict, train, evaluate, metrics, data, build_dataset,
                           adversarial, generation, transformer_clf, plagiarism, storage, visualization,
                           report, app (FastAPI) + web/ (UI), templates/
data/                      example dataset + dataset card (external/ and processed/ are git-ignored)
models/en/                 shipped model bundle + model card
reports/                   evaluation report, figures, example analyses
tests/                     pytest suite         notebooks/  evaluation explorer
scripts/                   data download, example dataset, evaluation report rendering
docs/architecture.md       design document      experiments/runs/  experiment logs
```

## Tests

```bash
python -m pytest            # whole suite (includes optional transformer tests if torch is installed)
```

## Configuration

`config/config.yaml`: host/port, privacy (history text storage off by default), device (`auto` / `cpu` /
`gpu`), feature profile, window size, calibration methods, band target (share of human sentence words
allowed in the AI band), evidence criteria, confidence levels, hybrid category definitions,
change-point settings and similarity settings. Values marked *derived* are replaced by measurements stored
in the model bundle.

## Licences and citations

Code: MIT. The example dataset is a CC BY 3.0 subset of the Ghostbuster data (attribution in
`data/README.md`). Downloaded corpora keep their own terms. Please cite: Wang et al., *M4: Multi-generator,
Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection* (EACL 2024); Verma et al.,
*Ghostbuster: Detecting Text Ghostwritten by Large Language Models* (NAACL 2024); Liang et al., *GPT
detectors are biased against non-native English writers* (Patterns, 2023).
