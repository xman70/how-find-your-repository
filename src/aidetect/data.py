"""Dataset construction: loaders, metadata schema, artefact cleaning, leakage-safe
splitting, near-duplicate removal, hybrid (spliced) documents, length augmentation.

Sample schema (one JSON object per line in ``*.jsonl``)::

    id               unique id
    text             normalised text
    label            0 = human, 1 = AI (binary target; hybrids: 1 if ai_fraction >= 0.5)
    ai_fraction      share of sentences/characters written by an AI system (0..1)
    hybrid_class     0 human, 1 human-assisted, 2 AI-assisted, 3 AI-generated
    author_type      human | ai | human_edited_ai | ai_assisted_human | paraphrased_ai |
                     machine_translated | hybrid_spliced
    generator        generator model ("none" for human text)
    generator_family vendor/model family used for leave-one-generator-out tests
    model_version    exact version if known, else "unknown"
    prompt_type      how the text was elicited (e.g. "title->abstract", "continue", "question->answer")
    domain           source domain (reddit, wikipedia, arxiv, essay, ...)
    genre            academic | informal | student | professional | scientific | news | creative | ...
    language         ISO 639-1
    n_words          word count after cleaning
    generation_date  year or "unknown"
    editing_level    none | light | heavy | ai_polished | unknown
    paraphrasing_level none | tool | manual | unknown
    writer_group     native | non_native | unknown (used only for fairness evaluation)
    source_dataset   M4 | Ghostbuster | Liang2023 | local | example
    group_id         documents sharing a group (same prompt / same source text) never cross splits
    segments         optional [[start, end, label], ...] character spans for hybrid documents
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .preprocessing import count_words, normalize_text, trim_incomplete_tail

HYBRID_NAMES = {0: "Human", 1: "Human-assisted", 2: "AI-assisted", 3: "AI-generated", 4: "Unknown"}

DOMAIN_GENRE = {
    "wikipedia": "encyclopedic (highly edited)",
    "wikihow": "instructional",
    "reddit": "informal",
    "arxiv": "scientific",
    "peerread": "academic review",
    "essay": "student",
    "reuter": "news (professional)",
    "wp": "creative",
    "bawe": "academic (university students)",
    "ets": "student (non-native English)",
    "pelic": "student (non-native English learners)",
    "lang8": "informal (non-native English learners)",
    "toefl": "student (non-native English)",
    "hewlett": "student (US 8th grade, native)",
    "cs224n": "academic (graduate students)",
    "college_essay": "student (college applications)",
    "legal": "legal",
}

# M4 file stem suffix -> (generator, family, model_version, approx generation year)
M4_GENERATORS = {
    "chatgpt": ("chatgpt", "openai", "gpt-3.5-turbo", "2023"),
    "davinci": ("davinci", "openai", "text-davinci-003", "2023"),
    "cohere": ("cohere", "cohere", "cohere-command (2023)", "2023"),
    "dolly": ("dolly", "databricks", "dolly-v2-12b", "2023"),
    "dolly2": ("dolly", "databricks", "dolly-v2-12b", "2023"),
    "bloomz": ("bloomz", "bigscience", "bloomz-176b", "2023"),
    "flant5": ("flant5", "google", "flan-t5", "2023"),
    "llama": ("llama", "meta", "llama (2023)", "2023"),
}
M4_PROMPT_TYPE = {
    "arxiv": "title->abstract", "peerread": "paper->review", "reddit": "question->answer",
    "wikihow": "title+headline->article", "wikipedia": "title->article",
}

_LABEL_LINE = re.compile(r"^\s*(prompt|essay|title|story|answer|response|article|abstract|summary|question)\s*:.*$",
                         re.IGNORECASE)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def hybrid_class_for(ai_fraction: float, cfg: dict | None = None) -> int:
    h = (cfg or {}).get("hybrid", {})
    if ai_fraction < h.get("human_max_share", 0.15):
        return 0
    if ai_fraction < h.get("human_assisted_max_share", 0.5):
        return 1
    if ai_fraction < h.get("ai_assisted_max_share", 0.85):
        return 2
    return 3


def strip_label_lines(text: str, max_lines: int = 3) -> str:
    """Remove leading 'Prompt: ...' / 'Essay:' style labels produced by data collection."""
    lines = text.split("\n")
    i = 0
    while i < min(max_lines, len(lines)) and (not lines[i].strip() or _LABEL_LINE.match(lines[i])):
        i += 1
    return "\n".join(lines[i:])


def clean_text(text: str, strip_labels: bool = False, prompt: str | None = None) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    if prompt:
        p, ts = prompt.strip(), text.strip()
        if p and ts.startswith(p):
            t = ts[len(p):]
    if strip_labels:
        t = strip_label_lines(t)
    t = normalize_text(t)
    t = trim_incomplete_tail(t)
    return t


def make_sample(text: str, label: int, *, source_dataset: str, domain: str, generator: str = "none",
                generator_family: str = "none", model_version: str = "unknown", author_type: str | None = None,
                prompt_type: str = "unknown", group_id: str | None = None, language: str = "en",
                generation_date: str = "unknown", editing_level: str = "none", paraphrasing_level: str = "none",
                writer_group: str = "unknown", ai_fraction: float | None = None, extra: dict | None = None) -> dict:
    ai_fraction = float(label) if ai_fraction is None else float(ai_fraction)
    s = {
        "id": f"{source_dataset}:{domain}:{generator}:{sha1(text)[:16]}",
        "text": text,
        "label": int(label),
        "ai_fraction": ai_fraction,
        "hybrid_class": hybrid_class_for(ai_fraction),
        "author_type": author_type or ("ai" if label else "human"),
        "generator": generator,
        "generator_family": generator_family,
        "model_version": model_version,
        "prompt_type": prompt_type,
        "domain": domain,
        "genre": DOMAIN_GENRE.get(domain, "unknown"),
        "language": language,
        "n_words": count_words(text),
        "generation_date": generation_date,
        "editing_level": editing_level,
        "paraphrasing_level": paraphrasing_level,
        "writer_group": writer_group,
        "source_dataset": source_dataset,
        "group_id": group_id or f"{source_dataset}:{sha1(text)[:16]}",
    }
    if extra:
        s.update(extra)
    return s


def _ok_length(text: str, min_words: int) -> bool:
    return count_words(text) >= min_words


# --------------------------------------------------------------------------- M4
def load_m4(m4_dir: str | os.PathLike, domains: Iterable[str] = ("arxiv", "peerread", "reddit", "wikihow", "wikipedia"),
            max_machine_per_file: int = 250, max_human_per_domain: int = 900, min_words: int = 40,
            seed: int = 0) -> list[dict]:
    """Load M4 (Wang et al., EACL 2024). Human texts are deduplicated per domain;
    group_id = hash of the paired human text so all generations for one prompt
    stay in the same split."""
    rng = random.Random(seed)
    m4_dir = Path(m4_dir)
    out: list[dict] = []
    humans: dict[str, dict[str, dict]] = defaultdict(dict)
    for domain in domains:
        for path in sorted(m4_dir.glob(f"{domain}_*.jsonl")):
            stem = path.stem.split("_", 1)[1].lower()
            if stem == "human" or stem not in M4_GENERATORS:
                continue
            gen, fam, version, year = M4_GENERATORS[stem]
            rows = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            rng.shuffle(rows)
            n_machine = 0
            for r in rows:
                pairs = _m4_pairs(domain, stem, r)
                for human, machine, prompt in pairs:
                    h = clean_text(human, strip_labels=True)
                    m = clean_text(machine, strip_labels=True, prompt=prompt)
                    if not h or not _ok_length(h, min_words):
                        continue
                    gid = f"m4:{domain}:{sha1(h[:400])[:16]}"
                    if h not in humans[domain]:
                        humans[domain][h] = make_sample(h, 0, source_dataset="M4", domain=domain, group_id=gid,
                                                        prompt_type="human-written")
                    if n_machine < max_machine_per_file and m and _ok_length(m, min_words):
                        out.append(make_sample(m, 1, source_dataset="M4", domain=domain, generator=gen,
                                               generator_family=fam, model_version=version, group_id=gid,
                                               prompt_type=M4_PROMPT_TYPE.get(domain, "unknown"),
                                               generation_date=year))
                        n_machine += 1
                if n_machine >= max_machine_per_file:
                    break
    for domain, hs in humans.items():
        vals = list(hs.values())
        rng.shuffle(vals)
        out.extend(vals[:max_human_per_domain])
    return out


def _m4_pairs(domain: str, stem: str, r: dict) -> list[tuple[str, str, str | None]]:
    prompt = r.get("prompt") if isinstance(r.get("prompt"), str) else None
    if domain == "peerread":
        if stem == "bloomz":
            hs, ms = r.get("human_reviews") or [], r.get("bloom_reviews") or []
        else:
            hs, ms = r.get("human_text") or [], r.get("machine_text") or []
        if isinstance(hs, str):
            hs = [hs]
        if isinstance(ms, str):
            ms = [ms]
        return [(h, m, None) for h, m in zip(hs[:2], ms[:2]) if isinstance(h, str) and isinstance(m, str)]
    if stem == "bloomz":
        human = r.get("abstract") if domain == "arxiv" else r.get("text")
        machine = r.get("machine_answer") if domain == "reddit" else r.get("machine_abstract")
        return [(human, machine, None)] if isinstance(human, str) and isinstance(machine, str) else []
    human, machine = r.get("human_text"), r.get("machine_text")
    if isinstance(human, str) and isinstance(machine, str):
        return [(human, machine, prompt)]
    return []


# --------------------------------------------------------------------------- Ghostbuster
GB_SOURCES = {
    "human": ("none", "none", "none", "human-written"),
    "gpt": ("chatgpt", "openai", "gpt-3.5-turbo", "prompt-from-human-text"),
    "claude": ("claude", "anthropic", "claude (2023)", "generated-prompt"),
    "gpt_prompt1": ("chatgpt", "openai", "gpt-3.5-turbo", "prompt-variant-1"),
    "gpt_prompt2": ("chatgpt", "openai", "gpt-3.5-turbo", "prompt-variant-2"),
    "gpt_semantic": ("chatgpt", "openai", "gpt-3.5-turbo", "semantic-variant"),
    "gpt_writing": ("chatgpt", "openai", "gpt-3.5-turbo", "writing-style-variant"),
}


def _gb_files(base: Path, domain: str, source: str) -> list[tuple[str, Path]]:
    d = base / domain / source
    if not d.exists():
        return []
    if domain == "reuter":
        return [(f"{p.parent.name}/{p.stem}", p) for p in sorted(d.glob("*/*.txt"))]
    return [(p.stem, p) for p in sorted(d.glob("*.txt"))]


def load_ghostbuster(gb_dir: str | os.PathLike, domains=("essay", "reuter", "wp"), max_per_source: int = 400,
                     max_per_variant: int = 100, max_human: int = 800, min_words: int = 40,
                     seed: int = 0) -> list[dict]:
    """Ghostbuster data (Verma et al., NAACL 2024; CC BY 3.0). human/i and gpt*/i share a
    prompt and therefore a group; Claude files are not index-paired."""
    rng = random.Random(seed)
    base = Path(gb_dir)
    out = []
    for domain in domains:
        for source, (gen, fam, version, ptype) in GB_SOURCES.items():
            files = _gb_files(base, domain, source)
            rng.shuffle(files)
            cap = max_per_variant if source.startswith("gpt_") else (max_human if source == "human" else max_per_source)
            n = 0
            for key, path in files:
                if n >= cap:
                    break
                raw = path.read_text(encoding="utf-8", errors="replace")
                t = clean_text(raw, strip_labels=source != "human")
                if not _ok_length(t, min_words):
                    continue
                gid = f"gb:{domain}:{key}" if source != "claude" else f"gb:{domain}:claude:{key}"
                out.append(make_sample(t, 0 if source == "human" else 1, source_dataset="Ghostbuster", domain=domain,
                                       generator=gen, generator_family=fam, model_version=version, prompt_type=ptype,
                                       group_id=gid, generation_date="2023" if source != "human" else "unknown"))
                n += 1
    return out


def load_ghostbuster_other(gb_dir: str | os.PathLike, max_per_set: int = 300, min_words: int = 40,
                           seed: int = 0) -> list[dict]:
    """Held-out human populations for the false-positive audit, plus humanizer-paraphrased AI text."""
    rng = random.Random(seed)
    base = Path(gb_dir) / "other"
    spec = {
        "bawe": ("bawe", 0, "human", "unknown"),
        "ets": ("ets", 0, "human", "non_native"),
        "pelic": ("pelic", 0, "human", "non_native"),
        "lang8": ("lang8", 0, "human", "non_native"),
        "toefl91": ("toefl", 0, "human", "non_native"),
        "undetectable": ("undetectable", 1, "paraphrased_ai", "unknown"),
    }
    out = []
    for folder, (domain, label, atype, wg) in spec.items():
        files = sorted((base / folder).glob("*.txt"))
        rng.shuffle(files)
        n = 0
        for p in files:
            if n >= max_per_set:
                break
            t = clean_text(p.read_text(encoding="utf-8", errors="replace"))
            if not _ok_length(t, min_words):
                continue
            extra = {"eval_set": f"gb_{folder}"}
            s = make_sample(t, label, source_dataset="Ghostbuster", domain=domain, author_type=atype,
                            writer_group=wg, generator="chatgpt" if label else "none",
                            generator_family="openai" if label else "none",
                            paraphrasing_level="tool" if folder == "undetectable" else "none",
                            group_id=f"gb:other:{folder}:{p.stem}", extra=extra)
            out.append(s)
            n += 1
    return out


def load_ghostbuster_perturbations(gb_dir: str | os.PathLike, originals: dict[str, dict],
                                   levels=("10",), max_per_type: int = 201) -> list[dict]:
    """Perturbed documents (perturb/<type>/<level>/<i>.txt). Labels are recovered by
    matching the unperturbed version (level 0) to the labelled corpus."""
    base = Path(gb_dir) / "perturb"
    out = []
    if not base.exists():
        return out
    for ptype_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        zero = ptype_dir / "0"
        if not zero.exists():
            continue
        for level in levels:
            lvl = ptype_dir / level
            if not lvl.exists():
                continue
            for p in sorted(lvl.glob("*.txt"))[:max_per_type]:
                orig_path = zero / p.name
                if not orig_path.exists():
                    continue
                key = sha1(clean_text(orig_path.read_text(encoding="utf-8", errors="replace"))[:300])
                src = originals.get(key)
                if src is None:
                    continue
                t = clean_text(p.read_text(encoding="utf-8", errors="replace"))
                if not t:
                    continue
                s = dict(src)
                s.update(text=t, id=f"perturb:{ptype_dir.name}:{level}:{p.stem}", n_words=count_words(t),
                         eval_set=f"perturb_{ptype_dir.name}_{level}", perturbation=ptype_dir.name,
                         perturbation_level=int(level), group_id=src["group_id"])
                out.append(s)
    return out


# --------------------------------------------------------------------------- Liang et al. 2023
LIANG_SETS = {
    "TOEFL_real_91": ("toefl", 0, "human", "non_native", "none"),
    "HewlettStudentEssay_real_88": ("hewlett", 0, "human", "native", "none"),
    "CS224N_real_145": ("cs224n", 0, "human", "unknown", "none"),
    "CollegeEssay_real_70": ("college_essay", 0, "human", "unknown", "none"),
    "TOEFL_gpt4polished_91": ("toefl", 1, "ai_assisted_human", "non_native", "ai_polished"),
    "HewlettStudentEssay_GPTsimplify_88": ("hewlett", 1, "ai_assisted_human", "native", "ai_polished"),
    "CS224N_gpt3_145": ("cs224n", 1, "ai", "unknown", "none"),
    "CS224N_gpt3PromptEng_145": ("cs224n", 1, "ai", "unknown", "none"),
    "CollegeEssay_gpt3_31": ("college_essay", 1, "ai", "unknown", "none"),
    "CollegeEssay_gpt3PromptEng_31": ("college_essay", 1, "ai", "unknown", "none"),
}


def load_liang(bias_dir: str | os.PathLike, min_words: int = 40) -> list[dict]:
    """Liang et al. (2023) 'GPT detectors are biased against non-native English writers'."""
    base = Path(bias_dir) / "Data_and_Results"
    out = []
    for path in sorted(base.glob("*/*/data.json")):
        name = path.parent.name
        if name not in LIANG_SETS:
            continue
        domain, label, atype, wg, edit = LIANG_SETS[name]
        for i, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            t = clean_text(row.get("document", ""), strip_labels=bool(label))
            if not _ok_length(t, min_words):
                continue
            gen = "gpt-4" if "gpt4" in name else ("chatgpt" if "GPT" in name else ("gpt-3" if "gpt3" in name else "none"))
            out.append(make_sample(t, label, source_dataset="Liang2023", domain=domain, author_type=atype,
                                   writer_group=wg, editing_level=edit, generator=gen,
                                   generator_family="openai" if label else "none",
                                   ai_fraction=0.5 if atype == "ai_assisted_human" else None,
                                   group_id=f"liang:{name}:{i}", extra={"eval_set": f"liang_{name}"}))
    return out


def load_legal_texts(paths=("/usr/share/common-licenses",), chunk_words: int = 300) -> list[dict]:
    """Human-written legal text (software licences) split into ~300-word chunks."""
    out = []
    seen = set()
    for base in paths:
        for p in sorted(glob.glob(os.path.join(base, "*"))):
            if os.path.isdir(p) or os.path.islink(p):
                continue
            try:
                raw = Path(p).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            t = normalize_text(raw)
            if sha1(t[:2000]) in seen:
                continue
            seen.add(sha1(t[:2000]))
            paras = [x for x in t.split("\n\n") if count_words(x) >= 8]
            buf, n = [], 0
            for para in paras:
                buf.append(para)
                n += count_words(para)
                if n >= chunk_words:
                    txt = "\n\n".join(buf)
                    out.append(make_sample(txt, 0, source_dataset="local", domain="legal",
                                           group_id=f"legal:{os.path.basename(p)}",
                                           extra={"eval_set": "legal_licences"}))
                    buf, n = [], 0
    return out


# --------------------------------------------------------------------------- local corpora (any language)
def load_local_jsonl(path: str | os.PathLike, language: str, min_words: int = 40) -> list[dict]:
    """Load a user-supplied corpus. Required fields: ``text`` and ``label`` (0 human, 1 AI). Optional fields
    follow the sample schema above (domain, generator, generator_family, group_id, author_type, ...).
    Lines with ``"eval_set"`` are treated as evaluation-only (false-positive audit / adversarial)."""
    out = []
    for i, row in enumerate(load_jsonl(path)):
        t = clean_text(row.get("text", ""), strip_labels=bool(row.get("label")))
        if not _ok_length(t, min_words):
            continue
        keep = {k: v for k, v in row.items() if k not in ("text", "label", "id", "n_words")}
        s = make_sample(t, int(row["label"]), source_dataset=row.get("source_dataset", "local"),
                        domain=row.get("domain", "local"), generator=row.get("generator", "none" if not row["label"] else "unknown"),
                        generator_family=row.get("generator_family", "none" if not row["label"] else "unknown"),
                        group_id=row.get("group_id") or f"local:{Path(path).stem}:{i}", language=language)
        s.update({k: v for k, v in keep.items() if k not in ("group_id",)})
        s["language"] = language
        out.append(s)
    return out


# --------------------------------------------------------------------------- splitting
def group_split(samples: list[dict], test_fraction: float, calib_fraction: float, seed: int = 0,
                force_test_groups: set[str] | None = None) -> dict[str, list[dict]]:
    """Split by ``group_id`` so related documents never cross splits.

    Groups are shuffled within (source_dataset, domain) strata so every domain
    appears in every split. ``force_test_groups`` are always placed in the test
    split (used for documents whose perturbed versions are test items).
    """
    force_test_groups = force_test_groups or set()
    rng = random.Random(seed)
    strata: dict[tuple, list[str]] = defaultdict(list)
    seen = set()
    for s in samples:
        key = (s["source_dataset"], s["domain"])
        if s["group_id"] not in seen:
            strata[key].append(s["group_id"])
            seen.add(s["group_id"])
    assign: dict[str, str] = {}
    for key, groups in sorted(strata.items()):
        groups = sorted(groups)
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_fraction))
        n_cal = int(round(n * calib_fraction))
        forced = [g for g in groups if g in force_test_groups]
        free = [g for g in groups if g not in force_test_groups]
        for g in forced:
            assign[g] = "test"
        n_test = max(0, n_test - len(forced))
        for i, g in enumerate(free):
            assign[g] = "test" if i < n_test else ("calib" if i < n_test + n_cal else "train")
    out = {"train": [], "calib": [], "test": []}
    for s in samples:
        s = dict(s)
        s["split"] = assign[s["group_id"]]
        out[s["split"]].append(s)
    return out


def _shingles(text: str, k: int = 5) -> set[int]:
    w = re.findall(r"\w+", text.lower())
    if len(w) < k:
        return {zlib.crc32(" ".join(w).encode())}
    return {zlib.crc32(" ".join(w[i:i + k]).encode()) for i in range(len(w) - k + 1)}


class MinHashLSH:
    """Small MinHash-LSH index for near-duplicate detection (Jaccard on word 5-shingles)."""

    def __init__(self, num_perm: int = 64, bands: int = 16, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.p = (1 << 61) - 1
        self.a = rng.integers(1, self.p, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, self.p, num_perm, dtype=np.uint64)
        self.bands = bands
        self.rows = num_perm // bands
        self.buckets: dict[tuple, list[int]] = defaultdict(list)
        self.sets: list[set[int]] = []

    def signature(self, sh: set[int]) -> np.ndarray:
        x = np.fromiter(sh, dtype=np.uint64)
        if x.size == 0:
            x = np.array([0], dtype=np.uint64)
        h = (np.outer(self.a, x) + self.b[:, None]) % np.uint64(self.p)
        return h.min(axis=1)

    def add(self, text: str) -> None:
        sh = _shingles(text)
        sig = self.signature(sh)
        idx = len(self.sets)
        self.sets.append(sh)
        for band in range(self.bands):
            self.buckets[(band, tuple(sig[band * self.rows:(band + 1) * self.rows].tolist()))].append(idx)

    def max_jaccard(self, text: str) -> float:
        sh = _shingles(text)
        sig = self.signature(sh)
        cands = set()
        for band in range(self.bands):
            cands.update(self.buckets.get((band, tuple(sig[band * self.rows:(band + 1) * self.rows].tolist())), []))
        best = 0.0
        for c in cands:
            o = self.sets[c]
            j = len(sh & o) / max(1, len(sh | o))
            best = max(best, j)
        return best


def remove_near_duplicates(splits: dict[str, list[dict]], threshold: float = 0.5) -> dict:
    """Drop exact duplicates everywhere and calib/test documents that nearly duplicate training text."""
    report = {"exact_duplicates_removed": 0, "near_duplicates_removed": {"calib": 0, "test": 0}}
    seen = set()
    for name in ("train", "calib", "test"):
        kept = []
        for s in splits[name]:
            h = sha1(s["text"])
            if h in seen:
                report["exact_duplicates_removed"] += 1
                continue
            seen.add(h)
            kept.append(s)
        splits[name] = kept
    index = MinHashLSH()
    for s in splits["train"]:
        index.add(s["text"])
    for name in ("calib", "test"):
        kept = []
        for s in splits[name]:
            if index.max_jaccard(s["text"]) >= threshold:
                report["near_duplicates_removed"][name] += 1
                continue
            kept.append(s)
        splits[name] = kept
    return report


# --------------------------------------------------------------------------- hybrids and augmentation
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9Α-Ω])")


def rough_sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n\n"):
        out.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return out


def build_hybrids(samples: list[dict], n: int, seed: int = 0, cfg: dict | None = None) -> list[dict]:
    """Splice human and AI sentences on the same topic into one document.

    Patterns: H->A, A->H, H->A->H, A->H->A. AI share is drawn uniformly from
    [0.1, 0.9]. Half of the documents put a paragraph break at the boundary,
    half join the parts inside a paragraph, so formatting does not reveal the
    boundary. Character-level ``segments`` give exact sentence labels.
    """
    rng = random.Random(seed)
    by_group: dict[str, dict[int, list[dict]]] = defaultdict(lambda: {0: [], 1: []})
    for s in samples:
        if s.get("author_type") in ("human", "ai") and s["n_words"] >= 120:
            by_group[s["group_id"]][s["label"]].append(s)
    paired = [g for g, d in by_group.items() if d[0] and d[1]]
    humans = [s for s in samples if s["label"] == 0 and s.get("author_type") == "human" and s["n_words"] >= 120]
    ais = [s for s in samples if s["label"] == 1 and s.get("author_type") == "ai" and s["n_words"] >= 120]
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 5 and humans and ais:
        attempts += 1
        if paired and rng.random() < 0.8:
            g = rng.choice(paired)
            h, a = rng.choice(by_group[g][0]), rng.choice(by_group[g][1])
        else:
            h, a = rng.choice(humans), rng.choice(ais)
            if h["domain"] != a["domain"]:
                continue
        hs, as_ = rough_sentences(h["text"]), rough_sentences(a["text"])
        if len(hs) < 4 or len(as_) < 4:
            continue
        share = rng.uniform(0.1, 0.9)
        total = min(len(hs) + len(as_), rng.randint(10, 24))
        n_ai = max(1, min(len(as_), round(total * share)))
        n_h = max(1, min(len(hs), total - n_ai))
        pattern = rng.choice(["HA", "AH", "HAH", "AHA"])
        hp, ap = hs[:n_h], as_[:n_ai]
        parts: list[tuple[list[str], int]] = []
        if pattern == "HA":
            parts = [(hp, 0), (ap, 1)]
        elif pattern == "AH":
            parts = [(ap, 1), (hp, 0)]
        elif pattern == "HAH":
            k = max(1, len(hp) // 2)
            parts = [(hp[:k], 0), (ap, 1), (hp[k:], 0)]
        else:
            k = max(1, len(ap) // 2)
            parts = [(ap[:k], 1), (hp, 0), (ap[k:], 1)]
        parts = [(p, lab) for p, lab in parts if p]
        sep = "\n\n" if rng.random() < 0.5 else " "
        text, segs, pos = "", [], 0
        for j, (p, lab) in enumerate(parts):
            chunk = " ".join(p)
            if j:
                text += sep
                pos += len(sep)
            segs.append([pos, pos + len(chunk), lab])
            text += chunk
            pos += len(chunk)
        n_ai_chars = sum(b - a_ for a_, b, lab in segs if lab == 1)
        frac = n_ai_chars / max(1, sum(b - a_ for a_, b, _ in segs))
        s = make_sample(text, int(frac >= 0.5), source_dataset=h["source_dataset"], domain=h["domain"],
                        generator=a["generator"], generator_family=a["generator_family"],
                        model_version=a["model_version"], author_type="hybrid_spliced", group_id=h["group_id"],
                        ai_fraction=frac, extra={"segments": segs, "splice_pattern": pattern,
                                                 "boundary_paragraph_break": sep == "\n\n"})
        s["hybrid_class"] = hybrid_class_for(frac, cfg)
        s["id"] = f"hybrid:{len(out)}:{s['id']}"
        out.append(s)
    return out


def truncate_words(text: str, n_words: int) -> str:
    """Cut at the sentence boundary closest to (and at least) ``n_words`` words."""
    sents = rough_sentences(text)
    out, count = [], 0
    for s in sents:
        out.append(s)
        count += count_words(s)
        if count >= n_words:
            break
    return " ".join(out)


def length_augment(samples: list[dict], seed: int = 0, min_words: int = 50) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for s in samples:
        if s["n_words"] < 150 or s.get("segments"):
            continue
        target = rng.randint(min_words, int(0.7 * s["n_words"]))
        t = truncate_words(s["text"], target)
        if count_words(t) < min_words:
            continue
        a = dict(s)
        a.update(text=t, id=s["id"] + f":trunc{target}", n_words=count_words(t), augmented="truncation")
        out.append(a)
    return out


def save_jsonl(samples: list[dict], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")


def load_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def describe(samples: list[dict]) -> dict:
    c = Counter((s["source_dataset"], s["domain"], s["label"]) for s in samples)
    gens = Counter(s["generator"] for s in samples if s["label"] == 1)
    return {
        "n": len(samples),
        "n_human": sum(1 for s in samples if s["label"] == 0),
        "n_ai": sum(1 for s in samples if s["label"] == 1),
        "by_source_domain_label": {f"{a}|{b}|{'ai' if l else 'human'}": v for (a, b, l), v in sorted(c.items())},
        "ai_by_generator": dict(sorted(gens.items())),
        "median_words": float(np.median([s["n_words"] for s in samples])) if samples else 0.0,
    }


def dataset_version(samples: list[dict]) -> str:
    return sha1("\n".join(sorted(s["id"] for s in samples)))[:12]
