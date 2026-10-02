"""Text preprocessing: normalisation, language identification, segmentation and
per-sentence linguistic statistics.

Every downstream feature is an aggregation of the per-sentence records built
here, so a document is parsed exactly once and any span of sentences (a
sliding window, a paragraph, the whole document) can be featurised cheaply.
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .lexicons import LANGID_STOPWORDS, LEXICONS, compiled_transitions, fold


class PreprocessingError(Exception):
    """Raised for inputs that cannot be analysed (with a user-facing message)."""


class ModelMissingError(PreprocessingError):
    pass


class UnsupportedLanguageError(PreprocessingError):
    pass


# --------------------------------------------------------------------------- normalisation
_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
    " ": " ", " ": " ", " ": " ", " ": " ", "\t": " ",
    "…": "...",
})
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
_NUM_RANGE = re.compile(r"(?<=\d)\s*[‒–—―]\s*(?=\d)")
_DASHES = re.compile(r"\s*(?:-{2,}|[‒–—―])\s*")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•●▪>#]+|\(?\d{1,2}[.)])\s+")
_MULTISPACE = re.compile(r" {2,}")


def normalize_text(text: str) -> str:
    """Remove encoding/formatting artefacts while keeping the writing itself.

    Formatting conventions (curly vs straight quotes, double spaces after full
    stops, ``--`` vs an em dash, markdown bullets) depend on the platform a text
    was collected from rather than on who wrote it; left in place they let a
    classifier learn the data source instead of the writing.
    """
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = text.translate(_QUOTE_MAP)
    text = _NUM_RANGE.sub("-", text)
    text = _DASHES.sub(" — ", text)
    # scraping debris: separators left behind when list headings were stripped (",, ", ".;", " ,")
    text = re.sub(r"(^|\n)[ ,;]+", r"\1", text)
    text = re.sub(r"([.!?])[;,]+(?=\s|$)", r"\1", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    text = re.sub(r"[ ]+([,;:])", r"\1", text)
    lines = []
    for line in text.split("\n"):
        line = _LIST_MARKER.sub("", line)
        line = _MULTISPACE.sub(" ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    blocks = [_unwrap_block(b) for b in re.split(r"\n\s*\n", text)]
    return "\n\n".join(b for b in blocks if b.strip()).strip()


_TERMINAL = tuple('.!?:;"\')]')


def _unwrap_block(block: str) -> str:
    """Re-join lines broken by fixed-width hard wrapping (e.g. arXiv abstracts)."""
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if len(lines) < 2:
        return block.strip()
    lengths = sorted(len(ln) for ln in lines[:-1])
    median = lengths[len(lengths) // 2]
    open_ends = sum(1 for ln in lines[:-1] if not ln.endswith(_TERMINAL))
    if len(lines) >= 3 and 50 <= median <= 100 and open_ends >= 0.5 * (len(lines) - 1):
        return " ".join(lines)
    out = [lines[0]]
    for ln in lines[1:]:
        prev = out[-1]
        if not prev.endswith(_TERMINAL) and (ln[:1].islower() or ln[:1].isdigit()):
            out[-1] = prev + " " + ln
        else:
            out.append(ln)
    return "\n".join(out)


def is_heading(par: str) -> bool:
    """Short line without sentence-final punctuation (section title, label)."""
    p = par.strip()
    if not p or "\n" in p:
        return False
    n = len(_WORD_RE.findall(p))
    return n <= 10 and not p.endswith(('.', '!', '?', '"', "'", ")", "..."))


def split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of paragraphs in normalised text."""
    spans = []
    sep = r"\n\s*\n" if re.search(r"\n\s*\n", text) else r"\n"
    pos = 0
    for m in re.finditer(sep, text):
        if text[pos:m.start()].strip():
            spans.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    return spans


_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def word_tokens(text: str) -> list[str]:
    """Fast regex word tokenizer (used where a full parse is unnecessary)."""
    return [fold(w) for w in _WORD_RE.findall(text)]


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


# --------------------------------------------------------------------------- language id
@dataclass
class LanguageGuess:
    code: str            # ISO 639-1 code, or "und"
    confidence: float
    supported: bool
    detail: str


def detect_language(text: str, supported: tuple[str, ...] = ("en", "el")) -> LanguageGuess:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return LanguageGuess("und", 0.0, False, "Too little text to identify the language.")
    greek = sum(1 for c in letters if "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿")
    latin = sum(1 for c in letters if c.isascii() or "À" <= c <= "ɏ")
    n = len(letters)
    if greek / n >= 0.5:
        code, conf = "el", greek / n
    elif latin / n >= 0.5:
        toks = word_tokens(text)[:3000]
        if not toks:
            return LanguageGuess("und", 0.0, False, "No words found.")
        scores = {lang: sum(t in sw for t in toks) / len(toks) for lang, sw in LANGID_STOPWORDS.items() if lang != "el"}
        code = max(scores, key=scores.get)
        best = scores[code]
        if best < 0.08:
            return LanguageGuess("und", best, False, "Language could not be identified reliably.")
        conf = min(1.0, best / 0.35)
    else:
        return LanguageGuess("und", 0.0, False, "Script not supported (only Latin-script English and Greek).")
    ok = code in supported
    detail = "" if ok else f"Detected language '{code}' is not supported by the trained models."
    return LanguageGuess(code, float(conf), ok, detail)


# --------------------------------------------------------------------------- spaCy
_SPACY_DEFAULT = {"en": "en_core_web_sm", "el": "el_core_news_sm"}


@lru_cache(maxsize=4)
def get_nlp(lang: str, model_name: str | None = None):
    model_name = model_name or _SPACY_DEFAULT.get(lang)
    if model_name is None:
        raise UnsupportedLanguageError(f"No spaCy pipeline configured for language '{lang}'.")
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - spaCy is a hard requirement
        raise ModelMissingError("spaCy is not installed. Run START.bat (or pip install -r requirements.txt).") from exc
    try:
        nlp = spacy.load(model_name, exclude=["ner", "lemmatizer"])
    except OSError as exc:
        raise ModelMissingError(
            f"The spaCy language model '{model_name}' is missing. Run START.bat again or install it with "
            f"'pip install -r requirements.txt'."
        ) from exc
    nlp.max_length = 2_000_000
    return nlp


# --------------------------------------------------------------------------- records
_CLAUSE_DEPS = {"ccomp", "xcomp", "advcl", "acl", "relcl", "csubj", "csubjpass", "parataxis"}
_SUBORD_DEPS = {"ccomp", "advcl", "acl", "relcl", "csubj", "csubjpass"}
_PASSIVE_DEPS = {"nsubjpass", "auxpass", "csubjpass"}
_OPENING_BY_POS = {
    "PRON": "pronoun", "DET": "determiner", "SCONJ": "subordinate", "ADP": "preposition", "ADV": "adverb",
    "VERB": "verb", "AUX": "verb", "NOUN": "noun", "PROPN": "noun", "CCONJ": "coordinator", "ADJ": "adjective",
    "NUM": "number", "PART": "particle", "INTJ": "interjection",
}
OPENING_CLASSES = ["transition", "pronoun", "determiner", "subordinate", "preposition", "adverb", "verb",
                   "noun", "coordinator", "adjective", "number", "particle", "interjection", "other"]
COUNT_KEYS = [
    "n_words", "n_alpha", "n_chars", "comma", "semicolon", "colon", "dash", "hyphen", "paren", "quote",
    "question", "exclaim", "ellipsis", "period", "contraction", "function", "articles", "pron1s", "pron1p",
    "pron2", "pron3", "cconj", "sconj", "long_words", "pos_ADJ", "pos_ADV", "pos_NOUN", "pos_VERB",
    "pos_AUX", "pos_PRON", "pos_PROPN", "pos_NUM", "pos_ADP", "pos_DET", "pos_CCONJ", "pos_SCONJ",
    "clauses", "subord", "passive", "coord", "fragment", "finite",
]
_CK_INDEX = {k: i for i, k in enumerate(COUNT_KEYS)}


@dataclass
class SentenceRecord:
    idx: int
    para: int
    para_initial: bool
    start: int
    end: int
    text: str
    words: list[str]                 # folded alphabetic word tokens
    word_lens: list[int]
    counts: np.ndarray               # vector aligned with COUNT_KEYS
    depth: int
    dep_dist: float
    end_punct: str
    punct_seq: list[str]
    opening: str
    first_word: str
    pos_seq: list[str]
    transitions: list[tuple[str, str, bool, bool]]  # (category, phrase, initial, followed_by_comma)
    surprisal: np.ndarray = field(default_factory=lambda: np.zeros(0))
    embedding: np.ndarray | None = None

    def c(self, key: str) -> float:
        return float(self.counts[_CK_INDEX[key]])


@dataclass
class Document:
    text: str
    language: str
    paragraphs: list[tuple[int, int]]
    sentences: list[SentenceRecord]
    truncated: bool = False
    K: np.ndarray | None = None      # sentence cosine-similarity matrix (set by the feature extractor)

    @property
    def n_words(self) -> int:
        return int(sum(s.c("n_words") for s in self.sentences))

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)


def _punct_type(tok, prev_tok, next_tok) -> list[str]:
    t = tok.text
    if t == "-" or (len(t) > 1 and set(t) == {"-"}):
        joined = not tok.whitespace_ and prev_tok is not None and not prev_tok.whitespace_
        return ["hyphen" if joined else "dash"]
    if t == "—":
        return ["dash"]
    if t.startswith("..."):
        return ["ellipsis"]
    out = []
    for ch in t:
        if ch == ",":
            out.append("comma")
        elif ch == ";":
            out.append("semicolon")
        elif ch == ":":
            out.append("colon")
        elif ch in "([{":
            out.append("paren")
        elif ch == '"':
            out.append("quote")
        elif ch == "?":
            out.append("question")
        elif ch == "!":
            out.append("exclaim")
        elif ch == ".":
            out.append("period")
    return out


def _depths(sent) -> tuple[int, float]:
    root_i = sent.root.i
    memo: dict[int, int] = {}
    dists = []
    for tok in sent:
        chain = []
        cur = tok
        while cur.i not in memo and cur.i != root_i and cur.head.i != cur.i:
            chain.append(cur.i)
            cur = cur.head
            if len(chain) > 200:
                break
        base = memo.get(cur.i, 0)
        for j, ti in enumerate(reversed(chain)):
            memo[ti] = base + j + 1
        if tok.i != root_i and not tok.is_punct:
            dists.append(abs(tok.i - tok.head.i))
    depth = max(memo.values()) if memo else 0
    return depth, (float(np.mean(dists)) if dists else 0.0)


def _match_transitions(ftoks: list[str], comma_after: list[bool], patterns) -> list[tuple[str, str, bool, bool]]:
    found = []
    i = 0
    n = len(ftoks)
    while i < n:
        hit = None
        for toks, cat, initial_only in patterns:
            L = len(toks)
            if i + L <= n and tuple(ftoks[i:i + L]) == toks:
                if initial_only and i != 0:
                    continue
                hit = (cat, " ".join(toks), i == 0, comma_after[i + L - 1])
                i += L
                break
        if hit:
            found.append(hit)
        else:
            i += 1
    return found


def _sentence_record(sent, idx, para_idx, para_initial, offset, lang, lex, patterns) -> SentenceRecord | None:
    toks = [t for t in sent if not t.is_space]
    if not toks:
        return None
    counts = np.zeros(len(COUNT_KEYS), dtype=np.float64)
    words, word_lens, punct_seq, pos_seq = [], [], [], []
    ftoks, comma_after = [], []
    has_finite = False
    passive = False
    for j, t in enumerate(toks):
        prev_t = toks[j - 1] if j else None
        nxt = toks[j + 1] if j + 1 < len(toks) else None
        pos = t.pos_
        if t.is_punct or pos in ("PUNCT", "SYM") and not any(ch.isalnum() for ch in t.text):
            for p in _punct_type(t, prev_t, nxt):
                counts[_CK_INDEX[p]] += 1
                if p != "hyphen":
                    punct_seq.append(p)
            continue
        low = t.text.lower().replace("’", "'")
        if low in lex["contractions"] or (lang == "en" and low == "'s" and pos in ("AUX", "VERB")):
            counts[_CK_INDEX["contraction"]] += 1
            ftoks.extend(fold(low).replace("'", " ").split())
            comma_after.extend([False] * len(fold(low).replace("'", " ").split()))
            continue
        if low in ("'s", "'"):
            continue
        sub = fold(low).replace("'", " ").split()
        if sub:
            ftoks.extend(sub)
            comma_after.extend([False] * (len(sub) - 1) + [bool(nxt is not None and nxt.text == ",")])
        pos_seq.append(pos)
        counts[_CK_INDEX["n_words"]] += 1
        if any(ch.isalpha() for ch in t.text):
            w = fold(t.text)
            words.append(w)
            word_lens.append(len(t.text))
            counts[_CK_INDEX["n_alpha"]] += 1
            if len(t.text) >= 7:
                counts[_CK_INDEX["long_words"]] += 1
            for key in ("function", "articles", "pron1s", "pron1p", "pron2", "pron3", "cconj", "sconj"):
                if w in lex[key]:
                    counts[_CK_INDEX[key]] += 1
        pk = "pos_" + pos
        if pk in _CK_INDEX:
            counts[_CK_INDEX[pk]] += 1
        dep = t.dep_.lower()
        base = dep.split(":")[0]
        if base in _CLAUSE_DEPS or (base == "conj" and pos in ("VERB", "AUX") and t.head.pos_ in ("VERB", "AUX")):
            counts[_CK_INDEX["clauses"]] += 1
        if base in _SUBORD_DEPS:
            counts[_CK_INDEX["subord"]] += 1
        if dep == "cc":
            counts[_CK_INDEX["coord"]] += 1
        if dep in _PASSIVE_DEPS or dep.endswith(":pass") or (pos in ("VERB", "AUX") and "Pass" in t.morph.get("Voice")):
            passive = True
        if "Fin" in t.morph.get("VerbForm") or t.tag_ in ("VBD", "VBP", "VBZ", "MD"):
            has_finite = True
    if counts[_CK_INDEX["n_words"]] == 0 and not punct_seq:
        return None
    counts[_CK_INDEX["clauses"]] += 1 if sent.root.pos_ in ("VERB", "AUX") else 0
    counts[_CK_INDEX["passive"]] = 1.0 if passive else 0.0
    counts[_CK_INDEX["finite"]] = 1.0 if has_finite else 0.0
    counts[_CK_INDEX["fragment"]] = 0.0 if has_finite else 1.0
    text = sent.text.strip()
    counts[_CK_INDEX["n_chars"]] = len(text)
    depth, dep_dist = _depths(sent)

    end = "none"
    for t in reversed(toks):
        if t.text in ('"', "'", ")", "]"):
            continue
        if t.text.startswith("..."):
            end = "ellipsis"
        elif t.text in (".", "?", "!"):
            end = {".": "period", "?": "question", "!": "exclaim"}[t.text]
        elif t.is_punct and t.text and t.text[-1] in ".?!":
            end = {".": "period", "?": "question", "!": "exclaim"}[t.text[-1]]
        break

    trans = _match_transitions(ftoks, comma_after, patterns)
    content = [t for t in toks if not t.is_punct]
    if trans and trans[0][2]:
        opening = "transition"
    elif content:
        opening = _OPENING_BY_POS.get(content[0].pos_, "other")
    else:
        opening = "other"
    first_word = fold(content[0].text) if content else ""
    start = offset + sent.start_char
    return SentenceRecord(
        idx=idx, para=para_idx, para_initial=para_initial, start=start, end=start + len(sent.text.rstrip()),
        text=text, words=words, word_lens=word_lens, counts=counts, depth=depth, dep_dist=dep_dist,
        end_punct=end, punct_seq=punct_seq, opening=opening, first_word=first_word, pos_seq=pos_seq,
        transitions=trans,
    )


def build_document(text: str, lang: str, max_words: int | None = None, normalized: bool = False,
                   spacy_model: str | None = None) -> Document:
    """Normalise, segment and parse ``text`` into a :class:`Document`."""
    if not normalized:
        text = normalize_text(text)
    truncated = False
    if max_words is not None and count_words(text) > max_words:
        # cut at the paragraph/sentence level after max_words words
        words_seen = 0
        cut = len(text)
        for m in _WORD_RE.finditer(text):
            words_seen += 1
            if words_seen >= max_words:
                nxt = re.search(r"[.!?]\s|\n", text[m.end():])
                cut = m.end() + (nxt.end() if nxt else 0)
                break
        text = text[:cut].strip()
        truncated = True
    nlp = get_nlp(lang, spacy_model)
    lex = LEXICONS[lang]
    patterns = compiled_transitions(lang)
    paragraphs = split_paragraphs(text)
    # Section headings / labels are not sentences; skip them unless the whole text is made of them.
    body = [(a, b) for a, b in paragraphs if not is_heading(text[a:b])]
    if body:
        paragraphs = body
    sentences: list[SentenceRecord] = []
    kept_paras = []
    para_texts = [text[a:b] for a, b in paragraphs]
    for (a, b), pdoc in zip(paragraphs, nlp.pipe(para_texts, batch_size=32)):
        p_idx = len(kept_paras)
        first = True
        added = False
        for sent in pdoc.sents:
            rec = _sentence_record(sent, len(sentences), p_idx, first, a, lang, lex, patterns)
            if rec is None:
                continue
            sentences.append(rec)
            first = False
            added = True
        if added:
            kept_paras.append((a, b))
    return Document(text=text, language=lang, paragraphs=kept_paras, sentences=sentences, truncated=truncated)


def trim_incomplete_tail(text: str) -> str:
    """Drop a trailing sentence fragment without terminal punctuation.

    Used only when building datasets: generated texts are frequently cut off by
    token limits; trimming both classes symmetrically prevents the classifier
    from learning "ends mid-sentence" as a label artefact.
    """
    t = text.rstrip()
    if not t or t[-1] in '.!?"\')]':
        return t
    m = list(re.finditer(r"[.!?][\"')\]]?(?=\s)", t))
    if not m:
        return t
    return t[: m[-1].end()].rstrip()


def sentence_count_quick(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))


def opening_distribution(records: list[SentenceRecord]) -> Counter:
    return Counter(r.opening for r in records)
