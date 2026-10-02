# False-positive optimisation experiment - pre-registered protocol

Written 2026-10-02 22:51 UTC, before either variant was evaluated.

Variants (identical code, data and seed):
* **A - baseline**: training pool only (M4 + Ghostbuster, no non-native learner writing).
* **B - non-native augmented**: A + half of the Ghostbuster ETS, PELIC and Lang-8 human documents
  (assigned by a hash of the document id).

Evaluation (same documents for both): the held-out test split, the adversarial sets, and the
false-positive audit with the B-training half removed, i.e. the unseen halves of ETS/PELIC/Lang-8,
both TOEFL sources (never trained on), BAWE, Hewlett, CS224N, college essays and legal texts.

Selection rule: choose **B** only if
1. its maximum false-positive rate over the unseen non-native groups (gb_ets, gb_pelic, gb_lang8,
   gb_toefl91, liang_TOEFL_real_91) is lower than A's, **and**
2. its test ROC-AUC is at most 0.005 lower and its test false-negative rate at most 2 percentage
   points higher than A's.
Otherwise ship **A**. Both results are reported either way.
