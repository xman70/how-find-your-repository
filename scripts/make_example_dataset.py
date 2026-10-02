"""Create the small committed example dataset (data/example/example_dataset.jsonl).

Only Ghostbuster documents are used because that corpus is licensed CC BY 3.0,
which permits redistribution with attribution. Documents come from the held-out
*test* split, so the example set never overlaps the training data of the
shipped model. Texts were normalised (see preprocessing.normalize_text).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidetect.data import build_hybrids, load_jsonl, save_jsonl  # noqa: E402

ATTRIBUTION = ("Ghostbuster data (Verma, Fleisig, Tomlin & Klein, 2024), https://github.com/vivek3141/ghostbuster-data, "
               "licensed CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/). Changes: text normalisation, "
               "label-line removal, truncation; hybrids are spliced from these documents.")


def main(n_per_cell: int = 6, seed: int = 7) -> None:
    rng = random.Random(seed)
    test = [s for s in load_jsonl(ROOT / "data/processed/en/test.jsonl") if s["source_dataset"] == "Ghostbuster"]
    picked = []
    for domain in ("essay", "reuter", "wp"):
        for gen in ("none", "chatgpt", "claude"):
            cell = [s for s in test if s["domain"] == domain and s["generator"] == gen and 150 <= s["n_words"] <= 600
                    and s["prompt_type"] in ("human-written", "prompt-from-human-text", "generated-prompt")]
            rng.shuffle(cell)
            k = n_per_cell * (2 if gen == "none" else 1)
            picked.extend(cell[:k])
    hybrids = build_hybrids(picked, 6, seed=seed)
    out = []
    for s in picked + hybrids:
        s = dict(s)
        s["license"] = "CC BY 3.0"
        s["attribution"] = ATTRIBUTION
        s.pop("split", None)
        out.append(s)
    dest = ROOT / "data/example/example_dataset.jsonl"
    save_jsonl(out, dest)
    counts = {}
    for s in out:
        key = f"{s['domain']}|{s['author_type']}|{s['generator']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps(counts, indent=1))
    print(f"{len(out)} samples -> {dest}")


if __name__ == "__main__":
    main()
