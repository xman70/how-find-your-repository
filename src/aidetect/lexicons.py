"""Language-specific closed-class word lists used by the stylometric features.

All lookups use :func:`fold` (lower-case, diacritics removed, final sigma
normalised) so that Greek text with or without accents matches the same entries.
Lists are deliberately limited to function words and discourse markers: no
"AI vocabulary" lists are used, because such lists encode the fingerprint of
particular model versions and generalise poorly to unseen models.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache


@lru_cache(maxsize=200_000)
def fold(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("ς", "σ").replace("’", "'")


def _fset(words: str) -> frozenset[str]:
    return frozenset(fold(w) for w in words.split())


# --------------------------------------------------------------------------- English
EN = {
    "articles": _fset("a an the"),
    "pron1s": _fset("i me my mine myself"),
    "pron1p": _fset("we us our ours ourselves"),
    "pron2": _fset("you your yours yourself yourselves"),
    "pron3": _fset("he him his himself she her hers herself it its itself they them their theirs themselves"),
    "cconj": _fset("and but or nor yet so"),
    "sconj": _fset(
        "although though because since unless whereas while whilst whether if when whenever where "
        "wherever after before until till once than that"
    ),
    "function": _fset(
        "a an the i me my mine myself we us our ours ourselves you your yours yourself yourselves he him "
        "his himself she her hers herself it its itself they them their theirs themselves this that these "
        "those who whom whose which what and but or nor yet so although though because since unless whereas "
        "while whether if when where after before until once than of in to for with on at by from about into "
        "over under between through during without within against among across toward towards upon around "
        "behind beyond along despite except onto per via be is am are was were been being have has had do "
        "does did can could may might must shall should will would not no all any some each every both either "
        "neither such more most other another only also very just then now too as much many few own same "
        "there here up out off down again further"
    ),
    "contractions": frozenset({"n't", "'re", "'ve", "'ll", "'d", "'m"}),
}

EN_TRANSITIONS: dict[str, list[str]] = {
    "additive": [
        "furthermore", "moreover", "additionally", "in addition", "besides", "likewise", "similarly",
        "what is more", "equally important", "not only", "also",
    ],
    "contrastive": [
        "however", "nevertheless", "nonetheless", "on the other hand", "in contrast", "conversely",
        "on the contrary", "instead", "despite this", "that said", "even so", "still", "yet", "but",
        "although", "whereas", "by contrast",
    ],
    "causal": [
        "therefore", "thus", "hence", "consequently", "as a result", "accordingly", "because of this",
        "for this reason", "as such", "so",
    ],
    "conclusive": [
        "in conclusion", "to conclude", "in summary", "to summarize", "to summarise", "overall",
        "ultimately", "in short", "all in all", "to sum up", "in the end", "in essence",
    ],
    "sequential": [
        "first", "firstly", "second", "secondly", "third", "thirdly", "next", "then", "finally", "lastly",
        "subsequently", "meanwhile", "afterwards", "afterward", "first of all",
    ],
    "exemplifying": [
        "for example", "for instance", "in particular", "specifically", "namely", "to illustrate",
        "such as",
    ],
    "emphatic": [
        "indeed", "in fact", "importantly", "notably", "crucially", "of course", "undoubtedly", "clearly",
        "it is important to note", "it is worth noting", "it's important to note", "it's worth noting",
    ],
}
# Single words that are only discourse markers in sentence-initial position.
EN_INITIAL_ONLY = frozenset({"also", "still", "yet", "but", "so", "then", "first", "second", "third", "next",
                             "although", "clearly", "overall", "finally", "instead"})

# --------------------------------------------------------------------------- Greek
EL = {
    "articles": _fset("ο η το οι τα του της των τον την τη τους τις ένας μία μια ένα ενός μιας έναν"),
    "pron1s": _fset("εγώ μου με εμένα εμένανε"),
    "pron1p": _fset("εμείς μας εμάς"),
    "pron2": _fset("εσύ σου σε εσένα εσείς σας εσάς"),
    "pron3": _fset(
        "αυτός αυτή αυτό αυτοί αυτές αυτά αυτού αυτής αυτών αυτόν αυτήν αυτούς τους τις τον την "
        "εκείνος εκείνη εκείνο εκείνοι εκείνες εκείνα"
    ),
    "cconj": _fset("και κι ή αλλά όμως ούτε μήτε είτε"),
    "sconj": _fset("αν εάν όταν επειδή γιατί διότι ότι πως που ώστε αφού μόλις καθώς ενώ αν και μολονότι προτού πριν"),
    "function": _fset(
        "ο η το οι τα του της των τον την τη τους τις ένας μία μια ένα ενός μιας έναν εγώ εσύ αυτός αυτή αυτό "
        "εμείς εσείς αυτοί αυτές αυτά μου σου μας σας με σε και κι ή αλλά όμως ούτε είτε αν εάν όταν επειδή "
        "γιατί διότι ότι πως που ώστε αφού μόλις καθώς ενώ στο στη στην στον στα στους στις στου στης από για "
        "προς χωρίς κατά μετά παρά αντί ως μέχρι έως πριν μεταξύ να θα δεν μην μη ας είναι ήταν είμαι είσαι "
        "είμαστε είστε έχει έχουν έχω είχε είχαν πολύ πιο όλα όλοι όλες κάθε κάποιος κάποια κάποιο τι ποιος "
        "ποια ποιο εκεί εδώ τώρα ήδη επίσης μόνο ακόμα ακόμη τόσο όπως"
    ),
    "contractions": frozenset(),
}

EL_TRANSITIONS: dict[str, list[str]] = {
    "additive": [
        "επιπλέον", "επίσης", "επιπροσθέτως", "επιπρόσθετα", "εξάλλου", "παράλληλα", "συγχρόνως",
        "όχι μόνο", "καθώς και", "ομοίως", "αντίστοιχα", "ακόμη", "ακόμα",
    ],
    "contrastive": [
        "ωστόσο", "όμως", "αντίθετα", "αντιθέτως", "εντούτοις", "εν τούτοις", "παρόλα αυτά",
        "παρ' όλα αυτά", "από την άλλη πλευρά", "από την άλλη", "αν και", "μολονότι", "αλλά",
    ],
    "causal": [
        "επομένως", "συνεπώς", "άρα", "γι' αυτό", "γι αυτό", "έτσι", "ως αποτέλεσμα", "κατά συνέπεια",
        "συνακόλουθα", "οπότε",
    ],
    "conclusive": [
        "συμπερασματικά", "εν κατακλείδι", "συνοψίζοντας", "συνολικά", "εν ολίγοις", "καταληκτικά",
        "για να συνοψίσουμε", "εν τέλει",
    ],
    "sequential": ["πρώτον", "δεύτερον", "τρίτον", "αρχικά", "έπειτα", "στη συνέχεια", "κατόπιν", "τελικά", "τέλος", "ύστερα"],
    "exemplifying": ["για παράδειγμα", "παραδείγματος χάριν", "π.χ.", "συγκεκριμένα", "ειδικότερα", "ιδιαίτερα"],
    "emphatic": [
        "πράγματι", "μάλιστα", "αξίζει να σημειωθεί", "είναι σημαντικό να", "σαφώς", "προφανώς",
        "αναμφίβολα", "ουσιαστικά",
    ],
}
EL_INITIAL_ONLY = frozenset({fold(w) for w in ["ακόμη", "ακόμα", "αλλά", "όμως", "έτσι", "τέλος", "αν και"]})

LEXICONS = {"en": EN, "el": EL}
TRANSITIONS = {"en": EN_TRANSITIONS, "el": EL_TRANSITIONS}
INITIAL_ONLY = {"en": EN_INITIAL_ONLY, "el": EL_INITIAL_ONLY}

# Short high-frequency word lists for language identification only.
LANGID_STOPWORDS = {
    "en": _fset("the of and to in is that it for was on are with as be this by not or have from but they at you"),
    "el": _fset("και το να η της του σε την με που για τα είναι ο από δεν οι θα των στο τον στην"),
    "fr": _fset("le la les de des et est un une que qui dans pour pas sur au avec ce il elle sont"),
    "de": _fset("der die das und ist nicht ein eine zu den mit von sich auf für dem des im auch es"),
    "es": _fset("el la los las de y que en un una es por con para no se del al lo como"),
    "it": _fset("il la le di e che è un una per non con del della sono si gli da al"),
    "pt": _fset("o a os as de e que em um uma é para com não do da por se na no"),
    "nl": _fset("de het een en van is dat niet op te zijn met voor aan er die ook"),
}


def compiled_transitions(lang: str) -> list[tuple[tuple[str, ...], str, bool]]:
    """Transition phrases as folded token tuples, longest first.

    Returns ``(tokens, category, initial_only)`` triples.
    """
    out = []
    init_only = INITIAL_ONLY.get(lang, frozenset())
    for cat, phrases in TRANSITIONS.get(lang, {}).items():
        for p in phrases:
            fp = fold(p)
            toks = tuple(fp.replace(",", " ").replace("'", " ").split())
            out.append((toks, cat, fp in init_only))
    out.sort(key=lambda t: -len(t[0]))
    return out
