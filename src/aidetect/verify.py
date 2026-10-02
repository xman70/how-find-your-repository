"""Installation check used by START.bat:  python -m aidetect.verify"""
from __future__ import annotations

import importlib
import sys

REQUIRED = ["numpy", "scipy", "pandas", "sklearn", "joblib", "yaml", "spacy", "wordfreq", "fastapi", "uvicorn",
            "multipart", "jinja2", "plotly", "matplotlib", "reportlab", "docx", "pypdf", "httpx"]
OPTIONAL = ["lightgbm", "torch", "transformers", "sentence_transformers", "nltk"]


def main() -> int:
    ok = True
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  [FAIL] Python 3.10 or newer is required")
        ok = False
    for m in REQUIRED:
        try:
            importlib.import_module(m)
            print(f"  [ok]   {m}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {m}: {exc}")
            ok = False
    for m in OPTIONAL:
        try:
            importlib.import_module(m)
            print(f"  [ok]   {m} (optional)")
        except Exception:  # noqa: BLE001
            print(f"  [--]   {m} (optional, not installed)")
    try:
        from .preprocessing import get_nlp

        for lang in ("en", "el"):
            get_nlp(lang)
            print(f"  [ok]   spaCy model for '{lang}'")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] spaCy language model: {exc}")
        ok = False
    try:
        from .device import resolve_device

        dev, note = resolve_device("auto")
        print(f"  [ok]   compute device: {dev}{' - ' + note if note else ''}")
        from .predict import Analyzer

        langs = Analyzer().available_languages()
        if langs:
            print(f"  [ok]   trained models: {', '.join(langs)}")
        else:
            print("  [WARN] no trained model found in models/ (run python train.py after building the dataset)")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] model check: {exc}")
        ok = False
    print("Installation OK" if ok else "Installation has problems (see FAIL lines above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
