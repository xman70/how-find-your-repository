"""Stylometric, punctuation, vocabulary, repetition, transition, syntactic,
predictability and burstiness features of a span of sentences.

Design notes
* Raw counts are converted to rates (per word / per sentence / per 100 words).
* Measures whose expected value depends on text length (type-token ratio,
  hapax ratio, n-gram repetition, entropy estimates) are computed on fixed-size
  chunks and averaged, so short and long texts are comparable.
* Undefined values (e.g. a standard deviation over one sentence) are NaN; the
  models impute them. No value is ever treated as evidence on its own.
"""
from __future__ import annotations

import math
import zlib
from collections import Counter

import numpy as np

from ..lexicons import LEXICONS
from ..preprocessing import OPENING_CLASSES, SentenceRecord
from .predictability import folded_frequencies, zipf

NAN = float("nan")
PUNCT_TYPES = ["comma", "semicolon", "colon", "dash", "paren", "quote", "question", "exclaim", "ellipsis", "period"]
END_TYPES = ["period", "question", "exclaim", "ellipsis", "none"]
TRANS_CATS = ["additive", "contrastive", "causal", "conclusive", "sequential", "exemplifying", "emphatic"]


# --------------------------------------------------------------------------- helpers
def _mean(x) -> float:
    return float(np.mean(x)) if len(x) else NAN


def _std(x) -> float:
    return float(np.std(x)) if len(x) > 1 else NAN


def _cv(x) -> float:
    if len(x) < 2:
        return NAN
    m = float(np.mean(x))
    return float(np.std(x) / m) if m > 0 else NAN


def burst_index(x) -> float:
    """Goh-Barabasi burstiness B = (sigma - mu) / (sigma + mu), in [-1, 1]."""
    if len(x) < 2:
        return NAN
    mu, sd = float(np.mean(x)), float(np.std(x))
    return (sd - mu) / (sd + mu) if sd + mu > 0 else NAN


def entropy(counts, k: int | None = None) -> float:
    vals = np.array([v for v in counts if v > 0], dtype=float)
    tot = vals.sum()
    if tot <= 0 or len(vals) == 0:
        return NAN
    p = vals / tot
    h = float(-(p * np.log(p)).sum())
    if k is not None and k > 1:
        h /= math.log(k)
    return h


def chunks(seq: list, size: int, min_last: int | None = None) -> list[list]:
    if not seq:
        return []
    min_last = size // 2 if min_last is None else min_last
    out = [seq[i:i + size] for i in range(0, len(seq), size)]
    if len(out) > 1 and len(out[-1]) < min_last:
        out[-2] = out[-2] + out[-1]
        out.pop()
    return out


def mattr(words: list[str], window: int = 50) -> float:
    n = len(words)
    if n == 0:
        return NAN
    if n <= window:
        return len(set(words)) / n
    counts = Counter(words[:window])
    types = len(counts)
    total = types
    for i in range(window, n):
        out_w, in_w = words[i - window], words[i]
        counts[out_w] -= 1
        if counts[out_w] == 0:
            types -= 1
        if counts[in_w] == 0:
            types += 1
        counts[in_w] += 1
        total += types
    return total / ((n - window + 1) * window)


def _mtld_pass(words: list[str], threshold: float) -> float:
    factors, types, count = 0.0, set(), 0
    for w in words:
        count += 1
        types.add(w)
        if len(types) / count <= threshold:
            factors += 1
            types, count = set(), 0
    if count > 0:
        ttr = len(types) / count
        factors += (1 - ttr) / (1 - threshold) if ttr < 1 else 0.0
    return len(words) / factors if factors > 0 else NAN


def mtld(words: list[str], threshold: float = 0.72) -> float:
    if len(words) < 50:
        return NAN
    a, b = _mtld_pass(words, threshold), _mtld_pass(words[::-1], threshold)
    vals = [v for v in (a, b) if not math.isnan(v)]
    return float(np.mean(vals)) if vals else NAN


def yule_k(words: list[str]) -> float:
    n = len(words)
    if n < 20:
        return NAN
    freq_of_freq = Counter(Counter(words).values())
    s2 = sum(i * i * v for i, v in freq_of_freq.items())
    return 1e4 * (s2 - n) / (n * n)


def ngram_repetition(tokens: list[str], n: int, chunk: int) -> float:
    """Share of n-gram occurrences that repeat an n-gram seen in the same chunk."""
    vals, weights = [], []
    for c in chunks(tokens, chunk):
        if len(c) < n + 2:
            continue
        grams = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1))
        tot = sum(grams.values())
        rep = sum(v for v in grams.values() if v > 1)
        vals.append(rep / tot)
        weights.append(tot)
    return float(np.average(vals, weights=weights)) if vals else NAN


def chunked_entropy(tokens: list, n: int, chunk: int, k: int) -> float:
    vals = []
    for c in chunks(tokens, chunk):
        if len(c) < n + 5:
            continue
        grams = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1))
        vals.append(entropy(grams.values(), k))
    return _mean(vals)


def chunked_type_ratio(tokens: list, n: int, chunk: int) -> float:
    vals = []
    for c in chunks(tokens, chunk):
        if len(c) < n + 5:
            continue
        grams = [tuple(c[i:i + n]) for i in range(len(c) - n + 1)]
        vals.append(len(set(grams)) / len(grams))
    return _mean(vals)


def block_share(values: list[str], block: int = 8) -> float:
    """Mean share of items that repeat another item in the same block of sentences."""
    if len(values) < 3:
        return NAN
    shares = []
    step = block if len(values) > block else len(values)
    for i in range(0, max(1, len(values) - step + 1), max(1, step // 2)):
        b = [v for v in values[i:i + step] if v]
        if len(b) < 3:
            continue
        c = Counter(b)
        shares.append(sum(1 for v in b if c[v] > 1) / len(b))
    return _mean(shares)


def compression_ratio(text: str, chunk_chars: int = 1000) -> float:
    text = text.lower()
    if len(text) < 300:
        return NAN
    parts = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    if len(parts) > 1 and len(parts[-1]) < 300:
        parts[-2] += parts[-1]
        parts.pop()
    ratios = [len(zlib.compress(p.encode("utf-8"), 9)) / len(p.encode("utf-8")) for p in parts]
    return float(np.mean(ratios))


# --------------------------------------------------------------------------- families
def stylometry(R: list[SentenceRecord], lang: str) -> dict[str, float]:
    sl = np.array([r.c("n_words") for r in R if r.c("n_words") > 0])
    wl = np.array([x for r in R for x in r.word_lens])
    tot = lambda k: sum(r.c(k) for r in R)  # noqa: E731
    W = max(tot("n_words"), 1.0)
    A = max(tot("n_alpha"), 1.0)
    nS = max(len(R), 1)
    f = {
        "sty_sent_len_mean": _mean(sl),
        "sty_sent_len_median": float(np.median(sl)) if len(sl) else NAN,
        "sty_sent_len_p10": float(np.percentile(sl, 10)) if len(sl) >= 3 else NAN,
        "sty_sent_len_p90": float(np.percentile(sl, 90)) if len(sl) >= 3 else NAN,
        "sty_short_sent_frac": float(np.mean(sl < 8)) if len(sl) else NAN,
        "sty_long_sent_frac": float(np.mean(sl > 30)) if len(sl) else NAN,
        "sty_word_len_mean": _mean(wl),
        "sty_word_len_std": _std(wl),
        "sty_long_word_frac": tot("long_words") / A,
        "sty_contraction_rate": 100 * tot("contraction") / W if lang == "en" else NAN,
        "sty_function_frac": tot("function") / A,
        "sty_article_frac": tot("articles") / A,
        "sty_pron1s_frac": tot("pron1s") / A,
        "sty_pron1p_frac": tot("pron1p") / A,
        "sty_pron2_frac": tot("pron2") / A,
        "sty_pron3_frac": tot("pron3") / A,
        "sty_cconj_frac": tot("cconj") / A,
        "sty_sconj_frac": tot("sconj") / A,
        "sty_question_frac": sum(r.end_punct == "question" for r in R) / nS,
        "sty_exclaim_frac": sum(r.end_punct == "exclaim" for r in R) / nS,
    }
    for pos in ("ADJ", "ADV", "NOUN", "VERB", "PROPN", "NUM", "PRON", "ADP", "DET", "AUX"):
        f[f"sty_pos_{pos.lower()}_frac"] = tot("pos_" + pos) / W
    return f


def punctuation(R: list[SentenceRecord]) -> dict[str, float]:
    tot = lambda k: sum(r.c(k) for r in R)  # noqa: E731
    W = max(sum(r.c("n_words") for r in R), 1.0)
    counts = {k: tot(k) for k in PUNCT_TYPES}
    f = {f"pun_{k}_rate": 100 * counts[k] / W for k in
         ("comma", "semicolon", "colon", "dash", "paren", "quote", "ellipsis")}
    f["pun_hyphen_rate"] = 100 * tot("hyphen") / W
    f["pun_total_rate"] = 100 * sum(counts.values()) / W
    f["pun_comma_per_sent_std"] = _std([r.c("comma") for r in R])
    f["pun_type_entropy"] = entropy(counts.values(), len(PUNCT_TYPES))
    f["pun_end_entropy"] = entropy(Counter(r.end_punct for r in R).values(), len(END_TYPES)) if len(R) >= 2 else NAN
    f["pun_no_end_frac"] = sum(r.end_punct == "none" for r in R) / max(len(R), 1)
    seq = [p for r in R for p in r.punct_seq]
    bigrams = Counter(zip(seq, seq[1:]))
    f["pun_transition_entropy"] = entropy(bigrams.values(), len(PUNCT_TYPES) ** 2) if sum(bigrams.values()) >= 3 else NAN
    return f


def vocabulary(R: list[SentenceRecord], lang: str, chunk: int = 150, window: int = 50) -> dict[str, float]:
    words = [w for r in R for w in r.words]
    func = LEXICONS[lang]["function"]
    content = [w for w in words if w not in func]
    freqs = folded_frequencies(lang)
    z = np.array([zipf(w, freqs) for w in content])
    f = {
        "voc_mattr": mattr(words, window),
        "voc_mtld": mtld(words),
        "voc_yule_k": yule_k(words),
        "voc_lexical_density": len(content) / len(words) if words else NAN,
        "voc_zipf_mean": float(z[z > 0].mean()) if (z > 0).any() else NAN,
        "voc_zipf_std": float(z[z > 0].std()) if (z > 0).sum() > 1 else NAN,
        "voc_rare_frac": float(np.mean((z > 0) & (z < 3.0))) if len(z) else NAN,
        "voc_oov_frac": float(np.mean(z == 0)) if len(z) else NAN,
    }
    hap, dis = [], []
    for c in chunks(words, 100):
        if len(c) < 50:
            continue
        cnt = Counter(c)
        types = len(cnt)
        hap.append(sum(1 for v in cnt.values() if v == 1) / types)
        dis.append(sum(1 for v in cnt.values() if v == 2) / types)
    f["voc_hapax_ratio"] = _mean(hap)
    f["voc_dis_ratio"] = _mean(dis)
    # local repetition of content words within the previous 50 tokens
    if len(content) >= 20:
        rep = 0
        for i, w in enumerate(content):
            if w in content[max(0, i - 50):i]:
                rep += 1
        f["voc_local_repeat_frac"] = rep / len(content)
    else:
        f["voc_local_repeat_frac"] = NAN
    # Heaps-law growth exponent (does vocabulary keep growing naturally?)
    if len(words) >= 120:
        seen, pts = set(), []
        for i, w in enumerate(words, 1):
            seen.add(w)
            if i % 10 == 0:
                pts.append((math.log(i), math.log(len(seen))))
        x, y = np.array(pts).T
        f["voc_heaps_slope"] = float(np.polyfit(x, y, 1)[0])
    else:
        f["voc_heaps_slope"] = NAN
    return f


def repetition(R: list[SentenceRecord], chunk: int = 150) -> dict[str, float]:
    words = [w for r in R for w in r.words]
    pos = [p for r in R for p in r.pos_seq]
    f = {
        "rep_bigram": ngram_repetition(words, 2, chunk),
        "rep_trigram": ngram_repetition(words, 3, chunk),
        "rep_4gram": ngram_repetition(words, 4, chunk),
        "rep_pos_trigram": ngram_repetition(pos, 3, chunk),
        "rep_sentence_template": block_share(["-".join(r.pos_seq[:3]) for r in R if len(r.pos_seq) >= 3]),
        "rep_opening_word": block_share([r.first_word for r in R]),
        "rep_compression": compression_ratio(" ".join(r.text for r in R)),
    }
    # repeated 5-word phrases per 100 words, within 300-word chunks
    vals = []
    for c in chunks(words, 300):
        if len(c) < 60:
            continue
        g = Counter(tuple(c[i:i + 5]) for i in range(len(c) - 4))
        vals.append(100 * sum(1 for v in g.values() if v > 1) / len(c))
    f["rep_phrase5_per100"] = _mean(vals)
    return f


def transitions(R: list[SentenceRecord]) -> dict[str, float]:
    T = [t for r in R for t in r.transitions]
    nS = max(len(R), 1)
    n = len(T)
    f = {
        "tra_per_sent": n / nS,
        "tra_sent_frac": sum(1 for r in R if r.transitions) / nS,
        "tra_initial_frac": sum(t[2] for t in T) / n if n else NAN,
        "tra_comma_frac": sum(t[3] for t in T) / n if n else NAN,
        "tra_diversity": len({t[1] for t in T}) / n if n >= 2 else NAN,
        "tra_category_entropy": entropy(Counter(t[0] for t in T).values(), len(TRANS_CATS)) if n >= 2 else NAN,
    }
    cats = Counter(t[0] for t in T)
    for c in TRANS_CATS:
        f[f"tra_{c}_rate"] = cats.get(c, 0) / nS
    idx = [i for i, r in enumerate(R) if r.transitions]
    f["tra_gap_cv"] = _cv(np.diff(idx)) if len(idx) >= 3 else NAN
    return f


def syntax(R: list[SentenceRecord], chunk: int = 150) -> dict[str, float]:
    nS = max(len(R), 1)
    depth = np.array([r.depth for r in R], dtype=float)
    pos = [p for r in R for p in r.pos_seq]
    tot = lambda k: sum(r.c(k) for r in R)  # noqa: E731
    return {
        "syn_depth_mean": _mean(depth),
        "syn_depth_std": _std(depth),
        "syn_dep_dist_mean": _mean([r.dep_dist for r in R if r.c("n_words") > 1]),
        "syn_clauses_per_sent": tot("clauses") / nS,
        "syn_subord_per_sent": tot("subord") / nS,
        "syn_coord_per_sent": tot("coord") / nS,
        "syn_passive_frac": tot("passive") / nS,
        "syn_fragment_frac": tot("fragment") / nS,
        "syn_opening_entropy": entropy(Counter(r.opening for r in R).values(), len(OPENING_CLASSES)) if len(R) >= 3 else NAN,
        "syn_pos_bigram_entropy": chunked_entropy(pos, 2, chunk, 17 * 17),
        "syn_pos_trigram_diversity": chunked_type_ratio(pos, 3, chunk),
    }


def predictability(R: list[SentenceRecord]) -> dict[str, float]:
    toks = np.concatenate([r.surprisal for r in R]) if R else np.zeros(0)
    sent = np.array([r.surprisal.mean() for r in R if len(r.surprisal) >= 3])
    return {
        "ppl_token_mean": _mean(toks),
        "ppl_token_median": float(np.median(toks)) if len(toks) else NAN,
        "ppl_token_q90": float(np.percentile(toks, 90)) if len(toks) >= 10 else NAN,
        "ppl_token_q10": float(np.percentile(toks, 10)) if len(toks) >= 10 else NAN,
        "ppl_sent_mean": _mean(sent),
        "ppl_sent_median": float(np.median(sent)) if len(sent) else NAN,
        "ppl_sent_min": float(sent.min()) if len(sent) else NAN,
        "ppl_sent_max": float(sent.max()) if len(sent) else NAN,
    }


def burstiness(R: list[SentenceRecord]) -> dict[str, float]:
    toks = np.concatenate([r.surprisal for r in R]) if R else np.zeros(0)
    sent = np.array([r.surprisal.mean() for r in R if len(r.surprisal) >= 3])
    sl = np.array([r.c("n_words") for r in R if r.c("n_words") > 0])
    sent_ttr = [len(set(r.words)) / len(r.words) for r in R if len(r.words) >= 5]
    depth = np.array([r.depth for r in R], dtype=float)
    roll = [np.mean(sent[i:i + 3]) for i in range(len(sent) - 2)] if len(sent) >= 4 else []
    return {
        "bur_ppl_sent_std": _std(sent),
        "bur_ppl_sent_cv": _cv(sent),
        "bur_ppl_sent_range": float(np.percentile(sent, 90) - np.percentile(sent, 10)) if len(sent) >= 3 else NAN,
        "bur_ppl_sent_B": burst_index(sent),
        "bur_ppl_token_std": _std(toks),
        "bur_ppl_rolling_std": _std(roll),
        "bur_sent_len_std": _std(sl),
        "bur_sent_len_cv": _cv(sl),
        "bur_sent_len_B": burst_index(sl),
        "bur_vocab_std": _std(sent_ttr),
        "bur_syntax_cv": _cv(depth),
    }
