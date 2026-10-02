import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidetect.config import load_config  # noqa: E402
from aidetect.data import group_split, load_jsonl, save_jsonl  # noqa: E402

EXAMPLE = ROOT / "data" / "example" / "example_dataset.jsonl"

HUMANISH = (
    "I wasn't sure what to expect when we drove up to the cabin last weekend. The road was a mess - mud everywhere, "
    "and my brother kept insisting we'd taken a wrong turn (we hadn't). Anyway, once we got there it was fine. "
    "Mostly fine. The stove didn't work, so dinner was cold beans and some bread that had gone a bit stale. "
    "Honestly? Best meal I've had in months. We stayed up talking about dad, about the old house, about nothing. "
    "Around two the rain started again and I lay there listening to it hammer the tin roof. I don't know why but "
    "I kept thinking about that summer when I was nine and fell off the dock. Funny what your brain digs up. "
    "Next morning the road had turned into a river, so we waited. Played cards. Lost, mostly."
)
AIISH = (
    "Climate change represents one of the most significant challenges facing humanity today. Furthermore, it "
    "affects ecosystems, economies, and communities across the globe. Rising temperatures contribute to more "
    "frequent extreme weather events, which in turn place considerable strain on infrastructure. Moreover, "
    "agricultural productivity is increasingly threatened by shifting precipitation patterns. In addition, "
    "coastal regions face heightened risks due to rising sea levels. Therefore, it is essential that governments, "
    "businesses, and individuals work together to implement effective mitigation strategies. Additionally, "
    "investing in renewable energy sources can significantly reduce greenhouse gas emissions. In conclusion, "
    "addressing climate change requires a comprehensive and coordinated approach that balances economic growth "
    "with environmental sustainability, ensuring a better future for generations to come."
)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def example_samples():
    return load_jsonl(EXAMPLE)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, example_samples, cfg):
    """Featurise the committed example dataset and train a small bundle with the real pipeline."""
    from aidetect.features import featurize_corpus
    from aidetect.train import train

    base = tmp_path_factory.mktemp("tiny")
    data_dir = base / "data"
    data_dir.mkdir()
    splits = group_split(example_samples, 0.0, 0.3, seed=1)
    for name, items in (("train", splits["train"]), ("calib", splits["calib"])):
        save_jsonl(items, data_dir / f"{name}.jsonl")
        feats = featurize_corpus(items, "en", "lite", cfg, n_jobs=1, windows_per_doc=8, seed=1, progress=False)
        with open(data_dir / f"features_{name}.pkl", "wb") as fh:
            pickle.dump(feats, fh)
    models = base / "models"
    train("en", data_dir=data_dir, out_path=models / "en" / "bundle.joblib", folds=2, max_windows=3000, verbose=False)
    return models


@pytest.fixture(scope="session")
def analyzer(tiny_model_dir, cfg):
    from aidetect.predict import Analyzer

    return Analyzer(config=cfg, device="cpu", models_dir=tiny_model_dir)


@pytest.fixture(scope="session")
def ok_result(analyzer, example_samples):
    text = next(s["text"] for s in example_samples if s["n_words"] > 250)
    res = analyzer.analyze(text, title="fixture")
    assert res["status"] == "ok", res
    return res
