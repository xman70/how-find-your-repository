"""Build and featurise every dataset used for training and evaluation.

    python -m aidetect.build_dataset --lang en --jobs 4

Outputs (data/processed/<lang>/):
    train/calib/test.jsonl         leakage-safe group splits of the training pool
    hybrids_<split>.jsonl          spliced human/AI documents with sentence labels
    train_augmented.jsonl          length-truncated copies of training documents
    fp_audit.jsonl                 held-out human populations (never trained on)
    adversarial.jsonl              perturbed / paraphrased / short / long / hybrid test items
    nonnative_train.jsonl          optional extra human data for the false-positive experiment
    features_<set>.pkl             featurised versions of the above
    dataset_card.json              counts, metadata coverage, leakage checks, dataset version
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np

from .adversarial import TRANSFORMS, synonym_available
from .config import DATA_DIR, load_config
from .data import (DOMAIN_GENRE, build_hybrids, clean_text, count_words, dataset_version, describe, group_split,
                   length_augment, load_ghostbuster, load_ghostbuster_other, load_ghostbuster_perturbations,
                   load_legal_texts, load_liang, load_local_jsonl, load_m4, make_sample, remove_near_duplicates,
                   save_jsonl, sha1,
                   truncate_words, _gb_files, GB_SOURCES)
from .features import featurize_corpus


def gb_originals_index(gb_dir: Path, domains=("essay", "reuter", "wp")) -> dict[str, dict]:
    """Map every labelled Ghostbuster document (cleaned text prefix hash) to sample metadata."""
    idx = {}
    for domain in domains:
        for source in ("human", "gpt", "claude"):
            gen, fam, version, ptype = GB_SOURCES[source]
            for key, path in _gb_files(gb_dir, domain, source):
                t = clean_text(path.read_text(encoding="utf-8", errors="replace"), strip_labels=source != "human")
                gid = f"gb:{domain}:{key}" if source != "claude" else f"gb:{domain}:claude:{key}"
                s = make_sample(t, 0 if source == "human" else 1, source_dataset="Ghostbuster", domain=domain,
                                generator=gen, generator_family=fam, model_version=version, prompt_type=ptype,
                                group_id=gid)
                idx[sha1(t[:300])] = s
    return idx


def local_adversarial(test: list[dict], n_per: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    base = [s for s in test if s.get("author_type") in ("human", "ai") and s["n_words"] >= 150]
    rng.shuffle(base)
    out = []
    for name, fn in TRANSFORMS.items():
        if name == "sim_wordnet_synonyms" and not synonym_available():
            print("  WordNet unavailable: skipping synonym-replacement set")
            continue
        for i, s in enumerate(base[:n_per]):
            t = fn(s["text"], seed=seed + i)
            a = dict(s)
            a.update(text=t, id=f"{name}:{s['id']}", n_words=count_words(t), eval_set=name, perturbation=name)
            out.append(a)
    # short texts
    for n_words in (50, 100):
        for s in base[:n_per]:
            t = truncate_words(s["text"], n_words)
            a = dict(s)
            a.update(text=t, id=f"short{n_words}:{s['id']}", n_words=count_words(t), eval_set=f"short_{n_words}_words")
            out.append(a)
    # long texts: concatenate 3 same-class, same-domain test documents
    pools: dict = {}
    for s in base:
        pools.setdefault((s["label"], s["domain"]), []).append(s)
    n_long = 0
    for (label, domain), items in sorted(pools.items()):
        for j in range(0, len(items) - 2, 3):
            if n_long >= n_per:
                break
            parts = items[j:j + 3]
            t = "\n\n".join(p["text"] for p in parts)
            a = dict(parts[0])
            a.update(text=t, id=f"long:{parts[0]['id']}", n_words=count_words(t), eval_set="long_concatenated",
                     group_id=parts[0]["group_id"])
            out.append(a)
            n_long += 1
    return out


def build_local(lang: str, local_paths: list[str], jobs: int = 4, out_dir: Path | None = None,
                skip_features: bool = False) -> dict:
    """Build a dataset for any language from user-supplied JSONL corpora (see data.load_local_jsonl)."""
    cfg = load_config()
    tc = cfg["training"]
    seed = int(tc["seed"])
    out_dir = out_dir or DATA_DIR / "processed" / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for p in local_paths:
        items.extend(load_local_jsonl(p, lang, tc["min_words_per_doc"]))
    pool = [s for s in items if not s.get("eval_set")]
    evaluation = [s for s in items if s.get("eval_set")]
    for s in pool:
        if s["n_words"] > tc["max_words_per_doc"]:
            s["text"] = truncate_words(s["text"], tc["max_words_per_doc"])
            s["n_words"] = count_words(s["text"])
    splits = group_split(pool, tc["test_fraction"], tc["calibration_fraction"], seed=seed)
    dup = remove_near_duplicates(splits, tc["near_duplicate_jaccard"])
    hyb = {k: build_hybrids(splits[k], n, seed=seed + i, cfg=cfg)
           for i, (k, n) in enumerate((("train", len(splits["train"]) // 6), ("calib", len(splits["calib"]) // 6),
                                       ("test", len(splits["test"]) // 6)))}
    sets = {"train": splits["train"], "calib": splits["calib"], "test": splits["test"],
            "hybrids_train": hyb["train"], "hybrids_calib": hyb["calib"], "hybrids_test": hyb["test"],
            "train_augmented": length_augment(splits["train"], seed=seed),
            "fp_audit": [s for s in evaluation if s["label"] == 0],
            "adversarial": [s for s in evaluation if s["label"] == 1] + local_adversarial(splits["test"], 100, seed)}
    for name, its in sets.items():
        save_jsonl(its, out_dir / f"{name}.jsonl")
    card = {"language": lang, "dataset_version": dataset_version(splits["train"] + splits["calib"] + splits["test"]),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed, "sources": local_paths,
            "leakage_controls": {"split_unit": "group_id", "deduplication": dup},
            "sets": {name: describe(its) for name, its in sets.items()}}
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    if not skip_features:
        featurize_sets(sets, lang, out_dir, jobs, cfg)
    return card


def build(lang: str = "en", jobs: int = 4, quick: bool = False, out_dir: Path | None = None,
          skip_features: bool = False) -> dict:
    cfg = load_config()
    tc = cfg["training"]
    seed = int(tc["seed"])
    ext = DATA_DIR / "external"
    out_dir = out_dir or DATA_DIR / "processed" / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = 0.15 if quick else 1.0
    t0 = time.time()
    print("Loading corpora ...", flush=True)
    m4 = load_m4(ext / "M4" / "data", max_machine_per_file=int(250 * scale), max_human_per_domain=int(900 * scale),
                 min_words=tc["min_words_per_doc"], seed=seed)
    gb_dir = ext / "ghostbuster-data"
    gb = load_ghostbuster(gb_dir, max_per_source=int(400 * scale), max_per_variant=int(100 * scale),
                          max_human=int(800 * scale),
                          min_words=tc["min_words_per_doc"], seed=seed)
    pool = m4 + gb
    for s in pool:  # cap very long documents at a sentence boundary
        if s["n_words"] > tc["max_words_per_doc"]:
            s["text"] = truncate_words(s["text"], tc["max_words_per_doc"])
            s["n_words"] = count_words(s["text"])
    print(f"  pool: {len(pool)} documents ({time.time() - t0:.0f}s)", flush=True)

    print("Indexing Ghostbuster originals for perturbation labels ...", flush=True)
    originals = gb_originals_index(gb_dir)
    perturbed = load_ghostbuster_perturbations(gb_dir, originals, levels=("10",), max_per_type=int(201 * scale) or 20)
    force_test = {s["group_id"] for s in perturbed}

    splits = group_split(pool, tc["test_fraction"], tc["calibration_fraction"], seed=seed, force_test_groups=force_test)
    dup = remove_near_duplicates(splits, tc["near_duplicate_jaccard"])
    train_groups = {s["group_id"] for s in splits["train"]} | {s["group_id"] for s in splits["calib"]}
    perturbed = [s for s in perturbed if s["group_id"] not in train_groups]
    print(f"  splits: { {k: len(v) for k, v in splits.items()} }  dedup: {dup}", flush=True)

    hyb = {
        "train": build_hybrids(splits["train"], int(1500 * scale), seed=seed + 1, cfg=cfg),
        "calib": build_hybrids(splits["calib"], int(500 * scale), seed=seed + 2, cfg=cfg),
        "test": build_hybrids(splits["test"], int(600 * scale), seed=seed + 3, cfg=cfg),
    }
    rng = random.Random(seed)
    aug_src = [s for s in splits["train"] if rng.random() < 0.5]
    augmented = length_augment(aug_src, seed=seed)

    print("Loading held-out populations (false-positive audit) and adversarial sets ...", flush=True)
    other = load_ghostbuster_other(gb_dir, max_per_set=int(300 * scale) or 30, seed=seed)
    liang = load_liang(ext / "ChatGPT-Detector-Bias")
    legal = load_legal_texts()
    fp_audit = [s for s in other if s["label"] == 0] + [s for s in liang if s["author_type"] == "human"] + legal
    # Non-native extra training data (FP-optimisation experiment): half of ETS/PELIC/Lang-8 by document;
    # the other half, TOEFL (both sources) and everything else stay unseen.
    nonnative_train = []
    for s in list(fp_audit):
        if s.get("eval_set") in ("gb_ets", "gb_pelic", "gb_lang8") and int(sha1(s["id"]), 16) % 2 == 0:
            nonnative_train.append(dict(s, split="train"))
    nn_ids = {s["id"] for s in nonnative_train}
    for s in fp_audit:
        s["nonnative_train_half"] = s["id"] in nn_ids

    adversarial = [s for s in other if s["label"] == 1]  # humanizer-paraphrased AI
    adversarial += [s for s in liang if s["author_type"] != "human"]
    adversarial += perturbed
    adversarial += local_adversarial(splits["test"], n_per=int(300 * scale) or 30, seed=seed)
    for h in hyb["test"]:
        a = dict(h)
        a["eval_set"] = "hybrid_spliced"
        adversarial.append(a)

    sets = {
        "train": splits["train"], "calib": splits["calib"], "test": splits["test"],
        "hybrids_train": hyb["train"], "hybrids_calib": hyb["calib"], "hybrids_test": hyb["test"],
        "train_augmented": augmented, "fp_audit": fp_audit, "adversarial": adversarial,
        "nonnative_train": nonnative_train,
    }
    for name, items in sets.items():
        save_jsonl(items, out_dir / f"{name}.jsonl")

    card = {
        "language": lang,
        "dataset_version": dataset_version(splits["train"] + splits["calib"] + splits["test"]),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "quick": quick,
        "sources": ["M4 (Wang et al., EACL 2024)", "Ghostbuster (Verma et al., NAACL 2024, CC BY 3.0)",
                    "Liang et al. 2023 (detector bias)", "local software licence texts (legal)"],
        "leakage_controls": {
            "split_unit": "group_id (shared prompt / shared human source text)",
            "deduplication": dup,
            "perturbation_originals_forced_to_test": len(force_test),
        },
        "sets": {name: describe(items) for name, items in sets.items()},
        "eval_set_counts": {},
        "genres": DOMAIN_GENRE,
    }
    from collections import Counter

    card["eval_set_counts"] = dict(Counter(s.get("eval_set", "?") for s in fp_audit + adversarial))
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"Datasets written to {out_dir} ({time.time() - t0:.0f}s)", flush=True)

    if not skip_features:
        featurize_sets(sets, lang, out_dir, jobs, cfg)
    return card


def featurize_sets(sets: dict, lang: str, out_dir: Path, jobs: int, cfg: dict) -> None:
    profile = cfg["training"]["profile"]
    wpd = int(cfg["training"]["windows_per_doc"])
    plan = {  # windows per document: sampled for training sets, all for evaluation sets
        "train": wpd, "calib": wpd, "hybrids_train": wpd * 2, "hybrids_calib": wpd * 2, "nonnative_train": wpd,
        "test": 10_000, "hybrids_test": 10_000, "fp_audit": 10_000, "adversarial": 10_000, "train_augmented": 0,
    }
    for name, items in sets.items():
        t0 = time.time()
        print(f"Featurising {name} ({len(items)} documents) ...", flush=True)
        res = featurize_corpus(items, lang, profile, cfg, n_jobs=jobs, windows_per_doc=plan.get(name, 0),
                               seed=int(cfg["training"]["seed"]), max_words=int(cfg["runtime"]["max_words"]))
        with open(out_dir / f"features_{name}.pkl", "wb") as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
        errs = sum(1 for r in res if "error" in r)
        print(f"  done in {time.time() - t0:.0f}s ({errs} errors)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="small subset for smoke tests")
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--local", nargs="*", default=None,
                    help="build from local JSONL corpora instead of the public English corpora (any language)")
    args = ap.parse_args(argv)
    if args.local:
        build_local(args.lang, args.local, args.jobs, skip_features=args.skip_features)
    elif args.lang != "en":
        raise SystemExit("Only English has built-in public corpora. For other languages pass --local corpus.jsonl ...")
    else:
        build(args.lang, args.jobs, args.quick, skip_features=args.skip_features)


if __name__ == "__main__":
    main()
