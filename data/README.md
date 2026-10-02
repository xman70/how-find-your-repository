# Data

This folder holds the dataset design, the small committed example dataset, and (git-ignored)
the downloaded research corpora and processed feature files.

```
data/
  example/example_dataset.jsonl   78 committed samples (CC BY 3.0 subset of Ghostbuster, test split only)
  external/                       downloaded corpora (git-ignored) - scripts/fetch_public_datasets.py
  processed/<lang>/               built splits + feature files (git-ignored) - python -m aidetect.build_dataset
  app.db                          local SQLite history / reference corpus (git-ignored)
```

## Sources

| Source | What it contributes | Licence | Used for |
|---|---|---|---|
| **M4** (Wang et al., EACL 2024) | Human text paired with outputs of **7 generator families** (ChatGPT, text-davinci-003, Cohere, Dolly-v2, BLOOMZ, Flan-T5, LLaMA) in 5 domains (arXiv abstracts, PeerRead reviews, Reddit ELI5, WikiHow, Wikipedia) | no licence file; research use, cited | training / calibration / test, cross-generator and cross-domain studies |
| **Ghostbuster** (Verma et al., NAACL 2024) | Student essays, Reuters news, creative writing (human / GPT-3.5 / Claude, plus 4 GPT prompt variants); non-native corpora (ETS, PELIC, Lang-8, TOEFL); British university writing (BAWE); humanizer-paraphrased AI text; 9 perturbation types | CC BY 3.0 | training, false-positive audit, adversarial tests, example dataset |
| **Liang et al. 2023** ("GPT detectors are biased against non-native English writers") | TOEFL essays (non-native), US 8th-grade essays (native), Stanford CS224N abstracts, college essays, GPT-4-polished / GPT-simplified human essays | no licence file; research use, cited | false-positive audit, AI-assisted-human tests |
| Software licence texts (`/usr/share/common-licenses` on Debian/Ubuntu) | Human-written legal text | each licence permits verbatim copying | false-positive audit (legal writing) |

Nothing from `external/` is committed. `scripts/fetch_public_datasets.py` downloads M4 and the
Liang et al. data over HTTPS and clones Ghostbuster with git (falls back to a zip download).

## Sample schema

Every sample (one JSON object per line) carries the metadata required for stratified
analysis: `id, text, label, ai_fraction, hybrid_class, author_type, generator,
generator_family, model_version, prompt_type, domain, genre, language, n_words,
generation_date, editing_level, paraphrasing_level, writer_group, source_dataset,
group_id`, plus `segments` (character spans with labels) for spliced hybrids and
`eval_set` for evaluation-only items. See `src/aidetect/data.py` for definitions.

`author_type` covers: `human`, `ai`, `hybrid_spliced`, `ai_assisted_human` (GPT-4-polished /
simplified human essays), `paraphrased_ai` (humanizer tool), and - when produced with
`aidetect.generation` - `ai_paraphrased_human`, `machine_translated` (incl. back-translation).

## Leakage controls

1. **Group-level splitting.** `group_id` = hash of the human source text that a prompt was
   derived from (M4), the shared prompt index (Ghostbuster `human/i` <-> `gpt*/i`), or the
   document itself. All texts sharing a group go to the same split
   (train 60 % / calibration 20 % / test 20 %, stratified by source and domain).
2. **Exact and near-duplicate removal.** Exact duplicates are dropped everywhere; calibration
   and test documents whose word-5-gram Jaccard similarity to any training document is >= 0.5
   (MinHash-LSH) are removed (the full build removed 1 exact and 18 near duplicates).
3. **Perturbation originals forced to test.** Ghostbuster perturbations are evaluated only
   when the unperturbed original is not in train/calibration (195 groups forced into test).
4. **Evaluation-only populations.** The false-positive audit sets (ETS, PELIC, Lang-8, TOEFL,
   BAWE, Hewlett, CS224N, college essays, legal) are never used for training, except in the
   explicitly labelled false-positive-optimisation experiment, which trains on half of
   ETS/PELIC/Lang-8 and evaluates only on the other half and the never-seen sets.

## Artefact cleaning (applied identically to both classes and at inference)

Measured collection artefacts that would otherwise let a classifier learn the *source* instead of
the *writing*: curly vs straight quotes, double spaces after full stops, `--` vs em dash,
markdown bullets/headings, fixed-width hard wraps (arXiv abstracts), scraping debris in
WikiHow human texts (7,251 `.,` separators vs 9 in machine text), echoed prompts and
`Prompt:`/`Essay:` labels in generated texts, generation cut-offs (trailing incomplete
sentences are trimmed from every text), section-heading lines. A single-feature audit after
cleaning found no feature separating the classes with AUC > 0.92 inside any domain.

## Difficult / borderline material

* Spliced human/AI hybrids on the **same topic** with exact sentence labels (2,600 documents;
  the boundary is a paragraph break in half of them and inside a paragraph in the other half).
* AI-polished and AI-simplified human essays (Liang et al.).
* Humanizer-paraphrased AI text (Ghostbuster `undetectable`).
* Character, word, sentence and paragraph perturbations incl. model paraphrasing (Ghostbuster).
* Local simulations: typo noise, sentence reordering, merge/split restructuring, casualisation,
  WordNet synonym replacement; truncated short texts; concatenated long texts.
* Length augmentation: truncated copies of half the training documents.

## Gaps (stated, not hidden)

* All AI text in the training data comes from **2023-era models**. Behaviour on newer models is
  unknown until data from them is added (`python -m aidetect.generation`).
* No public corpus of dyslexic writers was accessible, so that group is **not tested**.
* No genuine human post-edited AI text, translation, back-translation or temperature sweeps were
  available offline; the generation pipeline supports producing them.
* No Greek training/evaluation data was built (see README, "Greek support").

## Example dataset

`example/example_dataset.jsonl` - 78 samples (36 human, 36 AI from GPT-3.5 and Claude, 6 spliced
hybrids) drawn from the **held-out test split** of the Ghostbuster data, so it never overlaps the
shipped model's training data. Attribution: Ghostbuster data (Verma, Fleisig, Tomlin & Klein,
2024), https://github.com/vivek3141/ghostbuster-data, licensed CC BY 3.0
(https://creativecommons.org/licenses/by/3.0/). Changes: normalisation, label-line removal,
truncation; hybrids are spliced from these documents.
