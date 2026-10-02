"""Local adversarial transformations for robustness testing.

These are *simulations* of simple evasion edits; they are labelled as such in
every report. Real paraphrasing / humanizer output and model-based perturbations
come from the public Ghostbuster data (see data.py). Translation, back-
translation, decoding-temperature sweeps and genuine human post-editing require
generation backends (see generation.py) and are not simulated here.
"""
from __future__ import annotations

import random
import re

from .data import rough_sentences

_CONTRACT = [
    (r"\bdo not\b", "don't"), (r"\bdoes not\b", "doesn't"), (r"\bdid not\b", "didn't"), (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"), (r"\bcannot\b", "can't"), (r"\bcan not\b", "can't"), (r"\bwill not\b", "won't"),
    (r"\bit is\b", "it's"), (r"\bthat is\b", "that's"), (r"\bthey are\b", "they're"), (r"\bwe are\b", "we're"),
    (r"\byou are\b", "you're"), (r"\bI am\b", "I'm"), (r"\bthere is\b", "there's"), (r"\bwould not\b", "wouldn't"),
]


def typo_noise(text: str, rate: float = 0.03, seed: int = 0) -> str:
    rng = random.Random(seed)

    def mutate(m: re.Match) -> str:
        w = m.group(0)
        if len(w) < 4 or rng.random() > rate:
            return w
        i = rng.randrange(1, len(w) - 1)
        op = rng.choice(("swap", "drop", "double"))
        if op == "swap":
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if op == "drop":
            return w[:i] + w[i + 1:]
        return w[:i] + w[i] + w[i:]

    return re.sub(r"[A-Za-z]+", mutate, text)


def sentence_reorder(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    paras = []
    for para in text.split("\n\n"):
        s = rough_sentences(para)
        if len(s) > 2:
            head, rest = s[0], s[1:]
            rng.shuffle(rest)
            s = [head] + rest
        paras.append(" ".join(s))
    return "\n\n".join(paras)


def merge_split(text: str, seed: int = 0) -> str:
    """Merge adjacent short sentences and split long ones at clause boundaries."""
    rng = random.Random(seed)
    out_paras = []
    for para in text.split("\n\n"):
        sents = rough_sentences(para)
        res = []
        i = 0
        while i < len(sents):
            s = sents[i]
            if i + 1 < len(sents) and len(s.split()) < 14 and len(sents[i + 1].split()) < 14 and rng.random() < 0.6:
                nxt = sents[i + 1]
                s = s.rstrip(".!?") + ", and " + (nxt[0].lower() + nxt[1:] if nxt[:2] != "I " else nxt)
                i += 2
            else:
                i += 1
            if len(s.split()) > 28 and rng.random() < 0.7:
                m = re.search(r"(;|, which|, and|, but)\s", s[len(s) // 3:])
                if m:
                    cut = len(s) // 3 + m.start()
                    tail = s[cut + len(m.group(0)):].strip()
                    if tail:
                        s = s[:cut].rstrip(",;") + ". " + tail[0].upper() + tail[1:]
            res.append(s)
        out_paras.append(" ".join(res))
    return "\n\n".join(out_paras)


def casualize(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    t = text
    for pat, rep in _CONTRACT:
        t = re.sub(pat, lambda m, r=rep: r if rng.random() < 0.8 else m.group(0), t)
    t = re.sub(r";\s+(\w)", lambda m: ". " + m.group(1).upper(), t)
    t = re.sub(r"\s—\s", ", ", t)
    return t


_WN = None


def _wordnet():
    global _WN
    if _WN is None:
        try:
            from nltk.corpus import wordnet as wn

            wn.synsets("test")
            _WN = wn
        except LookupError:
            try:
                import nltk

                nltk.download("wordnet", quiet=True)
                from nltk.corpus import wordnet as wn

                wn.synsets("test")
                _WN = wn
            except Exception:  # noqa: BLE001
                _WN = False
        except Exception:  # noqa: BLE001
            _WN = False
    return _WN or None


def synonym_available() -> bool:
    return _wordnet() is not None


def synonym_replace(text: str, rate: float = 0.12, seed: int = 0) -> str:
    wn = _wordnet()
    if wn is None:
        raise RuntimeError("WordNet is not available (nltk wordnet corpus could not be loaded).")
    rng = random.Random(seed)
    stop = {"the", "and", "that", "with", "this", "from", "have", "were", "which", "their", "there", "been", "would"}

    def rep(m: re.Match) -> str:
        w = m.group(0)
        lw = w.lower()
        if len(w) < 5 or lw in stop or rng.random() > rate:
            return w
        cands = []
        for syn in wn.synsets(lw)[:3]:
            for lem in syn.lemmas()[:4]:
                name = lem.name()
                if "_" not in name and name.lower() != lw and name.isalpha():
                    cands.append(name)
        if not cands:
            return w
        c = rng.choice(cands)
        return c.capitalize() if w[0].isupper() else c

    return re.sub(r"[A-Za-z]+", rep, text)


TRANSFORMS = {
    "sim_typo_noise": typo_noise,
    "sim_sentence_reorder": sentence_reorder,
    "sim_merge_split_restructure": merge_split,
    "sim_casualize": casualize,
    "sim_wordnet_synonyms": synonym_replace,
}
