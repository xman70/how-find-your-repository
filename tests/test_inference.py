import socket

import pytest

from aidetect import DISCLAIMER
from aidetect.device import resolve_device
from aidetect.predict import Analyzer

from conftest import AIISH, HUMANISH


def test_result_structure(ok_result):
    r = ok_result["result"]
    total = r["ai_associated_share"] + r["human_associated_share"] + r["uncertain_share"]
    assert total == pytest.approx(1.0, abs=1e-6)
    for k in ("document_probability", "confidence", "estimated_ai_share"):
        assert 0.0 <= r[k] <= 1.0
    assert r["confidence_level"] in ("Low", "Medium", "High")
    assert r["hybrid"]["code"] in range(5)
    assert ok_result["disclaimer"] == DISCLAIMER
    assert ok_result["limitations"]
    sents = ok_result["sentences"]
    assert len(sents) == ok_result["meta"]["sentence_count"]
    assert all(0 <= s["probability"] <= 1 and s["confidence"] in ("Low", "Medium", "High") for s in sents)
    assert {d["name"] for d in ok_result["detectors"]} >= {"statistical", "neural", "perplexity"}
    assert "not proof" in ok_result["explanation"]["summary"]
    assert len(ok_result["series"]["sentence_probability"]) == len(sents)
    assert ok_result["paragraphs"] and ok_result["statistics"]["perplexity"]["items"]


def test_never_claims_certainty(ok_result):
    assert "definitely" not in ok_result["result"]["interpretation"].lower()
    assert 0.005 <= ok_result["result"]["document_probability"] <= 0.995


def test_too_short_text(analyzer):
    r = analyzer.analyze("Too short to analyse.")
    assert r["status"] == "insufficient_text"


def test_short_text_flags_insufficient_evidence(analyzer):
    r = analyzer.analyze(AIISH)
    assert r["status"] == "ok"
    assert r["result"]["evidence_level"] in ("insufficient", "low")
    if r["result"]["evidence_level"] == "insufficient":
        assert r["result"]["hybrid"]["label"] == "Unknown"


def test_unsupported_language(analyzer):
    fr = ("Le gouvernement a annoncé de nouvelles mesures pour les citoyens et la ville dans le pays. "
          "Les habitants de la région ont exprimé leur inquiétude face à ces changements importants. ") * 3
    r = analyzer.analyze(fr)
    assert r["status"] == "unsupported_language"


def test_greek_without_model_is_refused_not_misapplied(analyzer):
    el = ("Η κυβέρνηση ανακοίνωσε χθες νέα μέτρα για την οικονομία. Οι πολίτες περιμένουν να δουν τα αποτελέσματα "
          "στην καθημερινή τους ζωή, ενώ η αντιπολίτευση ασκεί έντονη κριτική. ") * 4
    r = analyzer.analyze(el)
    assert r["status"] == "model_missing"
    assert "Greek" in r["message"]


def test_missing_model_fails_gracefully(tmp_path, cfg):
    r = Analyzer(config=cfg, device="cpu", models_dir=tmp_path).analyze(HUMANISH * 2)
    assert r["status"] == "model_missing" and "train" in r["message"]


def test_gpu_unavailable_falls_back_to_cpu(monkeypatch):
    import aidetect.device as dev

    monkeypatch.setattr(dev, "cuda_available", lambda: False)
    d, note = resolve_device("gpu")
    assert d == "cpu" and "CPU" in note
    assert resolve_device("auto") == ("cpu", None)


def test_analysis_works_without_internet(analyzer, monkeypatch):
    def no_network(*a, **k):
        raise OSError("network disabled for this test")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    r = analyzer.analyze(HUMANISH + "\n\n" + AIISH)
    assert r["status"] == "ok"


def test_changepoints_are_labelled_as_potential(analyzer):
    r = analyzer.analyze(HUMANISH + "\n\n" + AIISH)
    for c in r["transitions"]["changepoints"]:
        assert c["label"] == "Potential authorship/style transition"
        assert set(c["differences"]) >= {"style_js", "perplexity_bits", "vocabulary_mattr", "syntax_depth", "semantic_distance"}
