"""Optional fine-tuned transformer classifier (PyTorch + Hugging Face transformers).

    python -m aidetect.transformer_clf --lang en --base-model distilroberta-base --epochs 2
    (Greek: --base-model xlm-roberta-base or a Greek BERT)

Trains on the training split only, then attaches itself to the existing model
bundle as an extra ensemble member: the stacker is refitted on calibration half
A and the calibrator on half B (see Ensemble.add_text_detector). Documents
longer than the context are split into chunks; the document score is the mean
chunk probability. Uses the GPU automatically when available.

Not run for the shipped model: the build environment could not download
pretrained weights. The code path is covered by a test that trains a tiny
randomly initialised model built locally.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .device import resolve_device


def _require():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("The transformer classifier needs torch and transformers "
                           "(pip install -r requirements-optional.txt).") from exc


class TransformerDetector:
    """Picklable handle: stores only the model directory; weights load lazily."""

    def __init__(self, model_dir: str | Path, device: str = "auto", max_length: int = 256, batch_size: int = 16):
        self.model_dir = str(model_dir)
        self.device_request = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.validation: dict = {}
        self._model = None
        self._tok = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_model"] = None
        d["_tok"] = None
        return d

    def _load(self):
        if self._model is None:
            _require()
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device, _ = resolve_device(self.device_request)
            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device).eval()

    def _chunks(self, text: str) -> list[str]:
        words = text.split()
        step = max(32, int(self.max_length * 0.6))
        return [" ".join(words[i:i + step]) for i in range(0, max(1, len(words)), step)] or [text]

    def predict_texts(self, texts: list[str]) -> np.ndarray:
        self._load()
        import torch

        out = []
        for t in texts:
            chunks = self._chunks(t)
            probs = []
            for i in range(0, len(chunks), self.batch_size):
                enc = self._tok(chunks[i:i + self.batch_size], truncation=True, max_length=self.max_length,
                                padding=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = self._model(**enc).logits
                probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
            out.append(float(np.mean(probs)))
        return np.array(out)


def train_transformer(texts: list[str], labels: list[int], base_model: str, out_dir: str | Path, epochs: int = 2,
                      lr: float = 3e-5, batch_size: int = 16, max_length: int = 256, device: str = "auto",
                      seed: int = 13, max_chunks_per_doc: int = 4) -> Path:
    _require()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    dev, _ = resolve_device(device)
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2).to(dev)
    helper = TransformerDetector(out_dir, max_length=max_length)
    pairs = []
    for t, y in zip(texts, labels):
        for c in helper._chunks(t)[:max_chunks_per_doc]:
            pairs.append((c, int(y)))
    n_pos = sum(y for _, y in pairs)
    w = torch.tensor([len(pairs) / (2 * max(1, len(pairs) - n_pos)), len(pairs) / (2 * max(1, n_pos))],
                     dtype=torch.float, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    model.train()
    for _ in range(epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc = tok([b[0] for b in batch], truncation=True, max_length=max_length, padding=True,
                      return_tensors="pt").to(dev)
            y = torch.tensor([b[1] for b in batch], device=dev)
            loss = loss_fn(model(**enc).logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "training.json").write_text(json.dumps({"base_model": base_model, "epochs": epochs, "lr": lr,
                                                       "n_chunks": len(pairs), "seed": seed}, indent=2))
    return out_dir


def attach(lang: str = "en", base_model: str = "distilroberta-base", epochs: int = 2, device: str = "auto",
           data_dir: Path | None = None, bundle_path: Path | None = None, max_train_docs: int | None = None) -> dict:
    import joblib

    from .config import DATA_DIR, MODELS_DIR
    from .train import doc_arrays, load_set

    data_dir = Path(data_dir or DATA_DIR / "processed" / lang)
    bundle_path = Path(bundle_path or MODELS_DIR / lang / "bundle.joblib")
    bundle = joblib.load(bundle_path)
    tr_s, tr_f = load_set(data_dir, "train")
    items = list(tr_s.values())[: max_train_docs or None]
    out_dir = bundle_path.parent / "transformer"
    train_transformer([s["text"] for s in items], [s["label"] for s in items], base_model, out_dir, epochs, device=device)
    det = TransformerDetector(out_dir, device=device)
    cs, cf = load_set(data_dir, "calib")
    names = bundle["doc"].feature_names
    X, y, g, meta = doc_arrays(cs, cf, names)
    texts = [m["text"] for m in meta]
    groups = sorted(set(g))
    half = set(groups[::2])
    a = np.array([gg in half for gg in g])
    rep = bundle["doc"].add_text_detector("transformer", det, X[a], [t for t, k in zip(texts, a) if k], y[a],
                                          X[~a], [t for t, k in zip(texts, ~a) if not k], y[~a])
    bundle["calibration_reports"]["document"] = rep
    bundle["training"]["transformer"] = {"base_model": base_model, "epochs": epochs}
    joblib.dump(bundle, bundle_path, compress=3)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--base-model", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--max-train-docs", type=int, default=None)
    args = ap.parse_args(argv)
    rep = attach(args.lang, args.base_model, args.epochs, args.device, max_train_docs=args.max_train_docs)
    print("Transformer attached. Calibration:", rep["selected"])


if __name__ == "__main__":
    main()
