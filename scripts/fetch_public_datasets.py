"""Download the public research corpora used for training and evaluation.

Nothing downloaded here is committed to the repository (data/external/ is
git-ignored). Please respect each dataset's licence and cite the authors:

* M4 (Wang et al., EACL 2024) - https://github.com/mbzuai-nlp/M4
  Multi-generator (ChatGPT, davinci, Cohere, Dolly, BLOOMZ, Flan-T5, LLaMA),
  multi-domain human/machine pairs. No licence file in the repository at the
  time of writing: research use, cite the paper.
* Ghostbuster data (Verma et al., NAACL 2024) - https://github.com/vivek3141/ghostbuster-data
  CC BY 3.0. Student essays, Reuters news, creative writing (human / GPT / Claude),
  non-native English corpora, perturbations, humanizer-paraphrased text.
* Liang et al. (2023) "GPT detectors are biased against non-native English writers"
  - https://github.com/Weixin-Liang/ChatGPT-Detector-Bias (TOEFL and US student essays).

Usage:  python scripts/fetch_public_datasets.py [--only m4,ghostbuster,liang]
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"

M4_FILES = [
    "arxiv_chatGPT", "arxiv_cohere", "arxiv_davinci", "arxiv_dolly", "arxiv_flant5", "arxiv_bloomz",
    "peerread_chatgpt", "peerread_cohere", "peerread_davinci", "peerread_dolly", "peerread_llama", "peerread_bloomz",
    "reddit_chatGPT", "reddit_cohere", "reddit_davinci", "reddit_dolly", "reddit_flant5", "reddit_bloomz",
    "wikihow_chatGPT", "wikihow_cohere", "wikihow_davinci", "wikihow_dolly2", "wikihow_bloomz",
    "wikipedia_chatgpt", "wikipedia_cohere", "wikipedia_davinci", "wikipedia_dolly", "wikipedia_bloomz",
]
REPOS = {
    "ghostbuster": ("https://github.com/vivek3141/ghostbuster-data", "master", "ghostbuster-data"),
    "liang": ("https://github.com/Weixin-Liang/ChatGPT-Detector-Bias", "main", "ChatGPT-Detector-Bias"),
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, length=1 << 20)
    tmp.replace(dest)


def fetch_m4() -> None:
    out = EXT / "M4" / "data"
    for name in M4_FILES:
        dest = out / f"{name}.jsonl"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [m4] {name} already present")
            continue
        url = f"https://raw.githubusercontent.com/mbzuai-nlp/M4/main/data/{name}.jsonl"
        print(f"  [m4] downloading {name} ...", flush=True)
        _download(url, dest)


def fetch_repo(key: str) -> None:
    url, branch, folder = REPOS[key]
    dest = EXT / folder
    if dest.exists() and any(dest.iterdir()):
        print(f"  [{key}] already present at {dest}")
        return
    if shutil.which("git"):
        print(f"  [{key}] git clone --depth 1 {url}", flush=True)
        r = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=False)
        if r.returncode == 0:
            return
        shutil.rmtree(dest, ignore_errors=True)
    zurl = f"{url}/archive/refs/heads/{branch}.zip"
    print(f"  [{key}] downloading {zurl}", flush=True)
    with urllib.request.urlopen(zurl, timeout=600) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(EXT)
    extracted = EXT / f"{folder}-{branch}"
    extracted.rename(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="m4,ghostbuster,liang")
    args = ap.parse_args(argv)
    EXT.mkdir(parents=True, exist_ok=True)
    wanted = {w.strip() for w in args.only.split(",") if w.strip()}
    try:
        if "m4" in wanted:
            fetch_m4()
        for key in ("ghostbuster", "liang"):
            if key in wanted:
                fetch_repo(key)
    except OSError as exc:
        print(f"Download failed: {exc}\nCheck your internet connection; nothing else on the system is affected.")
        return 1
    print(f"Done. Data in {EXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
