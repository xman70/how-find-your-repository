import io
import json

import httpx
import pytest

from aidetect import generation
from aidetect.io_utils import DocumentReadError, read_bytes
from aidetect.plagiarism import SimilarityChecker, exact_matches, ngram_containment, tokens, winnow
from aidetect.storage import Store

from conftest import AIISH, HUMANISH


# ---------------------------------------------------------------- similarity
def test_similarity_primitives():
    a, b = tokens(AIISH), tokens(HUMANISH + " " + AIISH)
    assert ngram_containment(a, b, 5) == 1.0
    assert ngram_containment(tokens(HUMANISH), tokens(AIISH), 5) < 0.05
    assert winnow(a) & winnow(b)
    assert exact_matches(a, b, 8)[0][2] >= len(a) - 1


def test_similarity_checker(tmp_path):
    chk = SimilarityChecker(Store(tmp_path / "s.db"))
    chk.add_reference("ref", AIISH)
    res = chk.check(HUMANISH + " " + AIISH)
    assert res["reference_corpus_size"] == 1 and res["sources"][0]["containment_5gram"] > 0.3
    assert 0.3 < res["overall_exact_overlap"] < 0.8
    assert chk.check(HUMANISH)["sources"] == []


# ---------------------------------------------------------------- document input
def test_read_txt_docx_pdf():
    assert "Ωστόσο" in read_bytes("Ωστόσο, κάτι.".encode("cp1253"), "greek.txt")
    import docx

    d = docx.Document()
    d.add_paragraph("Hello from a docx file.")
    buf = io.BytesIO()
    d.save(buf)
    assert "docx file" in read_bytes(buf.getvalue(), "a.docx")
    from reportlab.pdfgen import canvas

    pbuf = io.BytesIO()
    c = canvas.Canvas(pbuf)
    c.drawString(72, 720, "Hello from a PDF document.")
    c.save()
    assert "PDF document" in read_bytes(pbuf.getvalue(), "a.pdf")
    with pytest.raises(DocumentReadError):
        read_bytes(b"xx", "a.exe")
    with pytest.raises(DocumentReadError):
        read_bytes(b"not a pdf", "broken.pdf")


# ---------------------------------------------------------------- generation backends (mocked, no network)
def _transport(handler):
    return httpx.MockTransport(handler)


def test_openai_compatible_backend():
    def handler(req):
        body = json.loads(req.content)
        assert body["temperature"] == 0.3 and body["messages"][0]["role"] == "user"
        return httpx.Response(200, json={"model": "served-model", "choices": [{"message": {"content": "Generated text."},
                                                                                "finish_reason": "stop"}]})

    b = generation.OpenAICompatibleBackend("m", base_url="https://example.invalid/v1", transport=_transport(handler))
    g = b.generate("prompt", 0.3)
    assert g.text == "Generated text." and g.model == "served-model"


def test_gemini_and_ollama_backends():
    gem = generation.GeminiBackend("g-model", transport=_transport(lambda r: httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": "Gemini says hi."}]}, "finishReason": "STOP"}]})))
    assert gem.generate("p").text == "Gemini says hi."
    oll = generation.OllamaBackend("llama3.1", transport=_transport(lambda r: httpx.Response(
        200, json={"response": "Local model output.", "model": "llama3.1", "done_reason": "stop"})))
    assert oll.family == "llama" and not oll.external
    assert oll.generate("p").text == "Local model output."


class _Block:
    def __init__(self, t, text=""):
        self.type, self.text = t, text


class _Resp:
    def __init__(self, stop="end_turn", model="claude-opus-5-5", text="A generated essay. " * 30):
        self.stop_reason, self.model = stop, model
        self.content = [_Block("thinking"), _Block("text", text)]
        self.stop_details = None


class _FakeMessages:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def create(self, **kw):
        self.calls.append(kw)
        return self.resp


class _FakeClient:
    def __init__(self, resp):
        self.messages = _FakeMessages(resp)
        self.beta = type("B", (), {"messages": self.messages})()


def test_anthropic_backend_uses_fallbacks_and_records_served_model():
    fake = _FakeClient(_Resp(model="claude-opus-4-8"))
    b = generation.AnthropicBackend("claude-opus-5-5", effort="medium", client=fake)
    g = b.generate("prompt")
    call = fake.messages.calls[0]
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in call and call["output_config"] == {"effort": "medium"}
    assert g.model == "claude-opus-4-8"  # the model that actually served the request
    refused = generation.AnthropicBackend("claude-opus-5-5", client=_FakeClient(_Resp(stop="refusal")))
    with pytest.raises(generation.GenerationRefused):
        refused.generate("prompt")


def test_generate_dataset_metadata():
    src = {"id": "s1", "text": HUMANISH, "label": 0, "domain": "essay", "genre": "student", "group_id": "g1",
           "n_words": 160}
    b = generation.AnthropicBackend("claude-opus-5-5", client=_FakeClient(_Resp()))
    out = generation.generate_dataset(b, [src], ["polish"], 1, [None])
    s = out[0]
    assert s["author_type"] == "ai_assisted_human" and s["group_id"] == "g1" and s["generator_family"] == "anthropic"
    assert s["source_dataset"] == "generated" and s["editing_level"] == "ai_polished"


def test_external_upload_requires_consent(tmp_path, monkeypatch):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"id": "a", "text": "x", "label": 0}) + "\n")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    rc = generation.main(["--input", str(src), "--backend", "openai_compatible", "--model", "m",
                          "--out", str(tmp_path / "o.jsonl")])
    assert rc == 2 and not (tmp_path / "o.jsonl").exists()


# ---------------------------------------------------------------- experiment tracking
def test_tracking(tmp_path, monkeypatch):
    from aidetect import tracking

    monkeypatch.setattr(tracking, "RUNS_DIR", tmp_path)
    p = tracking.log_run("train", dataset_version="d", feature_version="f", model_version="m",
                         hyperparameters={"a": 1}, seed=3, validation={"auc": float("nan")})
    rec = json.loads(p.read_text())
    assert rec["seed"] == 3 and rec["validation"]["auc"] is None and rec["dataset_version"] == "d"
