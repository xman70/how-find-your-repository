## 9b. False-positive optimisation experiment (Phase 12)

Protocol written before evaluation: [`fp_optimization_protocol.md`](fp_optimization_protocol.md). Variant A = baseline; variant B = A + half of the ETS/PELIC/Lang-8 human essays. Same seed, code and test documents; the B-training half is excluded from every audit number. Threshold 0.5.

| Group | n | FPR A (baseline) | FPR B (shipped) | Seen by B? |
|---|---|---|---|---|
| gb_bawe | 300 | 3.0% | 2.3% | never |
| gb_ets | 146 | 50.7% | 8.9% | other half of same corpus |
| gb_lang8 | 139 | 28.1% | 3.6% | other half of same corpus |
| gb_pelic | 155 | 52.3% | 18.7% | other half of same corpus |
| gb_toefl91 | 91 | 44.0% | 11.0% | never |
| legal_licences | 99 | 16.2% | 13.1% | never |
| liang_CS224N_real_145 | 145 | 43.4% | 44.1% | never |
| liang_CollegeEssay_real_70 | 70 | 15.7% | 12.9% | never |
| liang_HewlettStudentEssay_real_88 | 88 | 34.1% | 2.3% | never |
| liang_TOEFL_real_91 | 91 | 44.0% | 9.9% | never |
| test_arxiv_scientific | 178 | 15.7% | 16.3% | in-distribution test |
| test_peerread_academic_reviews | 166 | 9.0% | 9.0% | in-distribution test |
| test_wikipedia_highly_edited | 189 | 14.8% | 16.4% | in-distribution test |
| test_reuters_professional_news | 168 | 4.2% | 4.2% | in-distribution test |
| test_student_essays | 180 | 11.1% | 10.0% | in-distribution test |

| Held-out test metric | A | B |
|---|---|---|
| roc_auc | 0.9748 | 0.9723 |
| accuracy | 0.9104 | 0.9067 |
| fpr | 0.0924 | 0.0917 |
| fnr | 0.0878 | 0.0944 |
| tpr_at_fpr_1pct | 0.7505 | 0.7171 |
| ece_balanced | 0.0206 | 0.0120 |

Selected AI-assisted / adversarial sets (AI detection rate):

| Set | A | B |
|---|---|---|
| liang_TOEFL_gpt4polished_91 | 76.9% | 68.1% |
| liang_HewlettStudentEssay_GPTsimplify_88 | 92.0% | 69.3% |
| liang_CollegeEssay_gpt3_31 | 87.1% | 77.4% |
| gb_undetectable | 95.0% | 95.0% |
| hybrid_spliced | 56.4% | 58.2% |

**Decision (by the pre-registered rule): ship B.** Maximum unseen non-native FPR 52.3% -> 18.7%; test ROC-AUC 0.9748 -> 0.9723 (-0.0025); FNR 8.78% -> 9.44%.

Interpretation: the strongest evidence is on the populations B never saw - both TOEFL sources (non-native) and the Hewlett essays (native US 8th graders) - where false positives fell from 44% to ~10% and from 34% to 2%. The cost is lower detection of human essays that GPT polished or simplified (an ambiguous, AI-assisted category). Graduate scientific abstracts (CS224N) remain at ~44% false positives under both variants: a serious, unresolved limitation.

