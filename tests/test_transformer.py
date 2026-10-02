"""Optional transformer path, exercised with a tiny randomly initialised model built locally
(no pretrained weights are downloaded). Skipped when torch/transformers are not installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from conftest import AIISH, HUMANISH  # noqa: E402


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tinybert")
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator([HUMANISH, AIISH] * 5, trainers.WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(d)
    cfg = BertConfig(vocab_size=fast.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=128, num_labels=2)
    BertForSequenceClassification(cfg).save_pretrained(d)
    return d


@pytest.mark.optional
def test_train_and_predict_tiny_transformer(tiny_base, tmp_path):
    from aidetect.transformer_clf import TransformerDetector, train_transformer

    texts = [HUMANISH, AIISH] * 6
    labels = [0, 1] * 6
    out = train_transformer(texts, labels, str(tiny_base), tmp_path / "clf", epochs=3, lr=5e-3, batch_size=4,
                            max_length=64, device="cpu")
    det = TransformerDetector(out, device="cpu", max_length=64)
    p = det.predict_texts([HUMANISH, AIISH])
    assert p.shape == (2,) and np.all((p >= 0) & (p <= 1))
    import pickle

    clone = pickle.loads(pickle.dumps(det))  # bundle stores only the path
    assert clone._model is None and np.allclose(clone.predict_texts([AIISH]), det.predict_texts([AIISH]))


@pytest.mark.optional
def test_ensemble_accepts_text_detector(tiny_base, tmp_path):
    from aidetect.models import Ensemble
    from aidetect.transformer_clf import TransformerDetector, train_transformer

    rng = np.random.default_rng(0)
    n = 80
    y = np.array([0, 1] * (n // 2))
    names = ["ppl_token_mean", "sty_sent_len_mean"]
    X = rng.normal(0, 1, (n, 2)) + y[:, None]
    texts = [HUMANISH if v == 0 else AIISH for v in y]
    ens = Ensemble("document", names, seed=0, folds=2,
                   detectors={"perplexity": ("logreg", ["perplexity"]), "stylometric": ("logreg", ["stylometric"])})
    ens.fit(X, y, np.arange(n), verbose=False)
    out = train_transformer(texts[:12], list(y[:12]), str(tiny_base), tmp_path / "clf", epochs=2, lr=5e-3,
                            batch_size=4, max_length=64, device="cpu")
    det = TransformerDetector(out, device="cpu", max_length=64)
    rep = ens.add_text_detector("transformer", det, X[:40], texts[:40], y[:40], X[40:], texts[40:], y[40:])
    assert rep["selected"] in ("platt", "isotonic")
    p = ens.predict_proba(X[:4], texts[:4])
    assert p.shape == (4,) and "transformer" in ens.weights
