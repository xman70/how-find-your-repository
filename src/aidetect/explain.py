"""Explanation engine: turns feature contributions into plain-language reasons.

Attribution method (model-agnostic, applied to the deployed ensemble): the
contribution of feature j is the change in the ensemble's logit when x_j is
replaced by its median in human-written training documents,

    c_j = logit f(x) - logit f(x with x_j := human median_j)

so c_j > 0 means "this measurement pushes the estimate toward AI-associated
writing relative to typical human writing". This is a single-reference,
one-feature-at-a-time approximation of Shapley values; it ignores interactions
and correlated features can share credit. Every explanation ends with the
caveat that these are statistical indicators, not proof.
"""
from __future__ import annotations

import math

import numpy as np

from . import SIGNAL_CAVEAT
from .features import family_of
from .models import Ensemble, percentile_in

# name -> (label, phrase when LOWER than typical human text, phrase when HIGHER)
FEATURE_TEXT: dict[str, tuple[str, str, str]] = {
    "ppl_token_mean": ("average word predictability", "unusually predictable word choices (low perplexity)", "less predictable word choices (higher perplexity)"),
    "ppl_token_median": ("median word predictability", "predominantly highly predictable words", "predominantly less predictable words"),
    "ppl_token_q90": ("rarest-word surprisal (90th pct.)", "few surprising word choices", "more surprising word choices"),
    "ppl_token_q10": ("most-predictable-word surprisal", "very common words dominate", "fewer very common words"),
    "ppl_sent_mean": ("average sentence perplexity", "low average sentence perplexity", "high average sentence perplexity"),
    "ppl_sent_median": ("median sentence perplexity", "low median sentence perplexity", "high median sentence perplexity"),
    "ppl_sent_min": ("lowest sentence perplexity", "some extremely predictable sentences", "no extremely predictable sentences"),
    "ppl_sent_max": ("highest sentence perplexity", "no highly surprising sentences", "some highly surprising sentences"),
    "bur_ppl_sent_std": ("sentence-to-sentence perplexity variation", "unusually uniform predictability between sentences (low burstiness)", "strongly varying predictability between sentences (high burstiness)"),
    "bur_ppl_sent_cv": ("relative perplexity variation", "low relative variation in sentence predictability", "high relative variation in sentence predictability"),
    "bur_ppl_sent_range": ("perplexity range across sentences", "narrow range of sentence predictability", "wide range of sentence predictability"),
    "bur_ppl_sent_B": ("perplexity burstiness index", "regular, non-bursty predictability", "bursty predictability"),
    "bur_ppl_token_std": ("word-level predictability variation", "low word-level variation in predictability", "high word-level variation in predictability"),
    "bur_ppl_rolling_std": ("local perplexity fluctuation", "smooth local predictability", "fluctuating local predictability"),
    "bur_sent_len_std": ("sentence-length variation", "highly uniform sentence lengths", "strongly varying sentence lengths"),
    "bur_sent_len_cv": ("relative sentence-length variation", "low relative variation in sentence length", "high relative variation in sentence length"),
    "bur_sent_len_B": ("sentence-length burstiness", "regular sentence-length rhythm", "bursty sentence-length rhythm"),
    "bur_vocab_std": ("vocabulary variation between sentences", "uniform vocabulary density across sentences", "varying vocabulary density across sentences"),
    "bur_syntax_cv": ("syntactic variation", "uniform syntactic complexity across sentences", "varying syntactic complexity across sentences"),
    "sty_sent_len_mean": ("average sentence length", "shorter sentences", "longer sentences"),
    "sty_sent_len_median": ("median sentence length", "shorter typical sentences", "longer typical sentences"),
    "sty_sent_len_p10": ("short-sentence length (10th pct.)", "very short sentences present", "no very short sentences"),
    "sty_sent_len_p90": ("long-sentence length (90th pct.)", "no very long sentences", "very long sentences present"),
    "sty_short_sent_frac": ("share of short sentences", "few short sentences", "many short sentences"),
    "sty_long_sent_frac": ("share of long sentences", "few long sentences", "many long sentences"),
    "sty_word_len_mean": ("average word length", "shorter words", "longer words"),
    "sty_word_len_std": ("word-length variation", "uniform word lengths", "varied word lengths"),
    "sty_long_word_frac": ("share of long words", "few long words", "many long words"),
    "sty_contraction_rate": ("contraction frequency", "few contractions (formal register)", "frequent contractions (informal register)"),
    "sty_function_frac": ("function-word share", "fewer function words", "more function words"),
    "sty_article_frac": ("article usage", "fewer articles", "more articles"),
    "sty_pron1s_frac": ("first-person singular pronouns", "little first-person singular voice", "frequent first-person singular voice"),
    "sty_pron1p_frac": ("first-person plural pronouns", "little use of 'we'", "frequent use of 'we'"),
    "sty_pron2_frac": ("second-person pronouns", "little direct address", "frequent direct address ('you')"),
    "sty_pron3_frac": ("third-person pronouns", "few third-person pronouns", "many third-person pronouns"),
    "sty_cconj_frac": ("coordinating conjunctions", "few coordinating conjunctions", "many coordinating conjunctions"),
    "sty_sconj_frac": ("subordinating conjunctions", "few subordinating conjunctions", "many subordinating conjunctions"),
    "sty_pos_adj_frac": ("adjective share", "few adjectives", "many adjectives"),
    "sty_pos_adv_frac": ("adverb share", "few adverbs", "many adverbs"),
    "sty_pos_noun_frac": ("noun share", "fewer nouns (less nominal style)", "more nouns (nominal, information-dense style)"),
    "sty_pos_verb_frac": ("verb share", "fewer verbs", "more verbs"),
    "sty_pos_propn_frac": ("proper-noun share", "few names and proper nouns", "many names and proper nouns"),
    "sty_pos_num_frac": ("number share", "few numbers and figures", "many numbers and figures"),
    "sty_pos_pron_frac": ("pronoun share", "few pronouns", "many pronouns"),
    "sty_pos_adp_frac": ("preposition share", "few prepositions", "many prepositions"),
    "sty_pos_det_frac": ("determiner share", "few determiners", "many determiners"),
    "sty_pos_aux_frac": ("auxiliary-verb share", "few auxiliary verbs", "many auxiliary verbs"),
    "sty_question_frac": ("questions", "few questions", "many questions"),
    "sty_exclaim_frac": ("exclamations", "few exclamations", "many exclamations"),
    "pun_comma_rate": ("comma density", "few commas", "many commas"),
    "pun_semicolon_rate": ("semicolon density", "few semicolons", "many semicolons"),
    "pun_colon_rate": ("colon density", "few colons", "many colons"),
    "pun_dash_rate": ("dash usage", "few dashes", "many dashes"),
    "pun_paren_rate": ("parenthesis usage", "few parentheses", "many parentheses"),
    "pun_quote_rate": ("quotation usage", "few quotation marks", "many quotation marks"),
    "pun_ellipsis_rate": ("ellipsis usage", "few ellipses", "many ellipses"),
    "pun_hyphen_rate": ("hyphenated compounds", "few hyphenated compounds", "many hyphenated compounds"),
    "pun_total_rate": ("overall punctuation density", "sparse punctuation", "dense punctuation"),
    "pun_comma_per_sent_std": ("comma-count variation", "uniform comma use per sentence", "varied comma use per sentence"),
    "pun_type_entropy": ("punctuation variety", "narrow punctuation repertoire", "broad punctuation repertoire"),
    "pun_end_entropy": ("sentence-ending variety", "uniform sentence endings", "varied sentence endings"),
    "pun_no_end_frac": ("unterminated sentences", "consistently terminated sentences", "sentences without final punctuation"),
    "pun_transition_entropy": ("punctuation sequence variety", "repetitive punctuation patterns", "varied punctuation patterns"),
    "voc_mattr": ("vocabulary diversity (MATTR)", "lower lexical diversity", "higher lexical diversity"),
    "voc_mtld": ("vocabulary diversity (MTLD)", "lower lexical diversity", "higher lexical diversity"),
    "voc_yule_k": ("vocabulary concentration (Yule's K)", "less concentrated vocabulary", "more concentrated, repetitive vocabulary"),
    "voc_lexical_density": ("lexical density", "fewer content words", "more content words"),
    "voc_zipf_mean": ("word commonness", "rarer, more sophisticated vocabulary", "very common vocabulary"),
    "voc_zipf_std": ("variation in word commonness", "uniform word commonness", "mix of common and rare words"),
    "voc_rare_frac": ("rare-word share", "few rare words", "many rare words"),
    "voc_oov_frac": ("unknown/misspelled word share", "few unknown words or misspellings", "many unknown words or misspellings"),
    "voc_hapax_ratio": ("words used only once", "fewer one-off words", "more one-off words"),
    "voc_dis_ratio": ("words used exactly twice", "fewer words used twice", "more words used twice"),
    "voc_local_repeat_frac": ("local word repetition", "little local repetition", "frequent local repetition"),
    "voc_heaps_slope": ("vocabulary growth rate", "vocabulary saturates quickly", "vocabulary keeps growing"),
    "syn_depth_mean": ("syntactic depth", "shallow sentence structure", "deeply nested sentence structure"),
    "syn_depth_std": ("syntactic depth variation", "uniform sentence structure depth", "varied sentence structure depth"),
    "syn_dep_dist_mean": ("dependency distance", "compact syntactic dependencies", "long-distance syntactic dependencies"),
    "syn_clauses_per_sent": ("clauses per sentence", "few clauses per sentence", "many clauses per sentence"),
    "syn_subord_per_sent": ("subordinate clauses", "few subordinate clauses", "many subordinate clauses"),
    "syn_coord_per_sent": ("coordination", "little coordination", "frequent coordination"),
    "syn_passive_frac": ("passive voice", "little passive voice", "frequent passive voice"),
    "syn_fragment_frac": ("sentence fragments", "few sentence fragments", "many sentence fragments"),
    "syn_opening_entropy": ("variety of sentence openings", "repetitive sentence openings", "varied sentence openings"),
    "syn_pos_bigram_entropy": ("part-of-speech variety", "repetitive grammatical sequences", "varied grammatical sequences"),
    "syn_pos_trigram_diversity": ("syntactic diversity", "low syntactic diversity", "high syntactic diversity"),
    "rep_bigram": ("repeated word pairs", "few repeated word pairs", "many repeated word pairs"),
    "rep_trigram": ("repeated three-word sequences", "few repeated phrases", "many repeated phrases"),
    "rep_4gram": ("repeated four-word sequences", "few repeated long phrases", "many repeated long phrases"),
    "rep_pos_trigram": ("repeated grammatical templates", "few repeated syntactic templates", "repeated syntactic templates"),
    "rep_sentence_template": ("repeated sentence structures", "varied sentence structures", "repeated sentence structures"),
    "rep_opening_word": ("repeated sentence-opening words", "varied first words", "repeated first words"),
    "rep_compression": ("compressibility", "highly redundant (compressible) text", "low redundancy"),
    "rep_phrase5_per100": ("repeated five-word phrases", "few repeated long phrases", "many repeated long phrases"),
    "tra_per_sent": ("transition-word frequency", "few transition words", "frequent transition words"),
    "tra_sent_frac": ("sentences with transitions", "few sentences with transitions", "many sentences with transitions"),
    "tra_initial_frac": ("sentence-initial transitions", "transitions placed mid-sentence", "transitions consistently placed at sentence starts"),
    "tra_comma_frac": ("transition followed by comma", "transitions rarely followed by commas", "transitions consistently followed by commas"),
    "tra_diversity": ("transition diversity", "the same transitions repeated", "varied transitions"),
    "tra_category_entropy": ("transition-type variety", "one kind of transition dominates", "many kinds of transitions"),
    "tra_additive_rate": ("additive transitions (moreover, furthermore)", "few additive transitions", "many additive transitions"),
    "tra_contrastive_rate": ("contrastive transitions (however)", "few contrastive transitions", "many contrastive transitions"),
    "tra_causal_rate": ("causal transitions (therefore)", "few causal transitions", "many causal transitions"),
    "tra_conclusive_rate": ("concluding transitions (in conclusion, overall)", "few concluding transitions", "many concluding transitions"),
    "tra_sequential_rate": ("sequencing transitions (first, finally)", "few sequencing transitions", "many sequencing transitions"),
    "tra_exemplifying_rate": ("exemplifying transitions (for example)", "few examples signposted", "many signposted examples"),
    "tra_emphatic_rate": ("emphatic markers (indeed, notably)", "few emphatic markers", "many emphatic markers"),
    "tra_gap_cv": ("spacing of transitions", "evenly spaced transitions", "irregularly spaced transitions"),
    "sem_adj_mean": ("similarity of neighbouring sentences", "weakly connected neighbouring sentences", "strongly connected neighbouring sentences"),
    "sem_adj_std": ("variation in sentence-to-sentence similarity", "uniform semantic progression", "uneven semantic progression"),
    "sem_adj_min": ("weakest sentence link", "abrupt topic jumps", "no abrupt topic jumps"),
    "sem_centroid_mean": ("topic concentration", "loosely focused topic", "tightly focused topic"),
    "sem_centroid_std": ("topic focus variation", "uniform topic focus", "varying topic focus"),
    "sem_redundancy": ("semantic redundancy", "little restatement", "frequent restatement of the same content"),
    "sem_local_gain": ("local coherence", "weak local coherence", "strong local coherence"),
    "sem_topic_jump_rate": ("topic jumps", "few topic jumps", "frequent topic jumps"),
    "sem_compression": ("semantic compression", "semantically diverse content", "semantically compressed content"),
    "style_drift_mean": ("style drift between sections", "stable style throughout", "style changes between sections"),
    "style_drift_max": ("largest style shift", "no pronounced style shift", "a pronounced style shift"),
    "style_drift_std": ("variation of style drift", "uniform style drift", "uneven style drift"),
    "cp_max_stat": ("strongest statistical change point", "no strong change point", "a strong statistical change point"),
}


def describe_feature(name: str) -> tuple[str, str, str]:
    if name in FEATURE_TEXT:
        return FEATURE_TEXT[name]
    label = name.split("_", 1)[-1].replace("_", " ")
    return label, f"lower {label}", f"higher {label}"


def human_baseline(ens: Ensemble) -> np.ndarray:
    out = np.full(len(ens.feature_names), np.nan)
    for j, n in enumerate(ens.feature_names):
        ref = ens.population.get(n, {}).get("human")
        if ref:
            out[j] = ref["quantiles"][len(ref["quantiles"]) // 2]
    return out


def _stack_logit(ens: Ensemble, X: np.ndarray, fixed: dict[str, np.ndarray] | None = None) -> np.ndarray:
    from .calibration import logit

    cols = []
    for n in ens.stack_names:
        if fixed and n in fixed:
            cols.append(fixed[n])
        else:
            cols.append(logit(ens.detectors[n].predict(X)["prob"]))
    return ens.stacker.decision_function(np.column_stack(cols))


def ablation_contributions(ens: Ensemble, X: np.ndarray, text_logits: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """(n_rows x n_features) logit contributions relative to the human-median reference."""
    n, F = X.shape
    base = human_baseline(ens)
    ref = _stack_logit(ens, X, text_logits)
    big = np.repeat(X, F, axis=0)
    rows = np.arange(n * F)
    cols = np.tile(np.arange(F), n)
    vals = np.tile(base, n)
    keep = ~np.isnan(vals)
    big[rows[keep], cols[keep]] = vals[keep]
    fixed = {k: np.repeat(v, F) for k, v in (text_logits or {}).items()}
    alt = _stack_logit(ens, big, fixed).reshape(n, F)
    return ref[:, None] - alt


def family_contributions(ens: Ensemble, contrib: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for j, n in enumerate(ens.feature_names):
        fam = family_of(n)
        out[fam] = out.get(fam, 0.0) + float(contrib[j])
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))


def factors_for_row(ens: Ensemble, x: np.ndarray, contrib: np.ndarray, top_k: int = 4,
                    min_abs: float = 0.05) -> dict:
    pop = ens.population
    order = np.argsort(-np.abs(contrib))
    toward_ai, toward_human = [], []
    for j in order:
        c = float(contrib[j])
        if abs(c) < min_abs or math.isnan(x[j]):
            continue
        name = ens.feature_names[j]
        label, low, high = describe_feature(name)
        ref = pop.get(name, {}).get("human")
        pct = percentile_in(ref["quantiles"], float(x[j])) if ref else float("nan")
        direction_low = ref is not None and x[j] < ref["mean"]
        text = low if direction_low else high
        item = {"feature": name, "family": family_of(name), "label": label, "contribution": round(c, 4),
                "value": float(x[j]), "human_percentile": None if math.isnan(pct) else round(pct, 1), "text": text}
        bucket = toward_ai if c > 0 else toward_human
        if any(b["text"] == text for b in bucket):
            continue  # e.g. two diversity measures that read the same in plain language
        bucket.append(item)
        if len(toward_ai) >= top_k and len(toward_human) >= max(1, top_k // 2):
            break
    return {"toward_ai": toward_ai[:top_k], "toward_human": toward_human[:max(1, top_k // 2)]}


def summary_text(factors: dict, elevated: bool = True) -> str:
    ai = factors.get("toward_ai", [])
    if not ai:
        return ("No individual signal stands out as strongly associated with AI-generated text. " + SIGNAL_CAVEAT)
    head = ("Factors contributing to the elevated score:" if elevated else
            "Strongest signals toward AI-associated patterns (the overall score is not elevated):")
    lines = "\n".join(f"• {f['text']}" for f in ai)
    return f"{head}\n{lines}\n{SIGNAL_CAVEAT}"
