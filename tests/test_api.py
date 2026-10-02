import io

import pytest
from fastapi.testclient import TestClient

from aidetect.app import create_app
from aidetect.storage import Store

from conftest import AIISH, HUMANISH


@pytest.fixture(scope="module")
def client(analyzer, cfg, tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("db") / "app.db")
    return TestClient(create_app(cfg, analyzer=analyzer, store=store))


def test_index_and_assets(client):
    r = client.get("/")
    assert r.status_code == 200 and "AI Text Analysis" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/vendor/plotly.min.js").status_code == 200


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "Local processing" in h["privacy"] and "en" in h["models"]


def test_analyze_text_history_and_reports(client):
    r = client.post("/api/analyze", json={"text": HUMANISH + "\n\n" + AIISH, "title": "api test"})
    assert r.status_code == 200
    res = r.json()
    assert res["status"] == "ok" and res["id"]
    hist = client.get("/api/analyses").json()
    assert any(a["id"] == res["id"] for a in hist)
    assert client.get(f"/api/analyses/{res['id']}").json()["result"] == res["result"]
    for fmt, ctype in (("pdf", "application/pdf"), ("html", "text/html"), ("json", "application/json")):
        rr = client.get(f"/api/analyses/{res['id']}/report.{fmt}")
        assert rr.status_code == 200 and rr.headers["content-type"].startswith(ctype)
    assert client.get(f"/api/analyses/{res['id']}/report.exe").status_code == 400
    assert client.delete(f"/api/analyses/{res['id']}").status_code == 200
    assert client.get(f"/api/analyses/{res['id']}").status_code == 404


def test_history_does_not_store_text_by_default(client, cfg):
    assert cfg["app"]["store_text_in_history"] is False


def test_analyze_files(client):
    import docx

    d = docx.Document()
    for para in (HUMANISH, AIISH):
        d.add_paragraph(para)
    buf = io.BytesIO()
    d.save(buf)
    r = client.post("/api/analyze-file", files={"file": ("essay.docx", buf.getvalue(),
                                                          "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert r.status_code == 200 and r.json()["status"] == "ok"
    r = client.post("/api/analyze-file", files={"file": ("a.txt", (HUMANISH * 2).encode("utf-8"), "text/plain")})
    assert r.json()["status"] == "ok"
    r = client.post("/api/analyze-file", files={"file": ("x.exe", b"MZ...", "application/octet-stream")})
    assert r.status_code == 400


def test_short_and_unsupported(client):
    assert client.post("/api/analyze", json={"text": "short"}).json()["status"] == "insufficient_text"


def test_similarity_is_separate(client):
    add = client.post("/api/references", json={"title": "source essay", "text": AIISH}).json()
    assert add["id"]
    assert client.post("/api/references", json={"title": "dup", "text": AIISH}).json()["duplicate"] is True
    refs = client.get("/api/references").json()
    assert any(r["title"] == "source essay" for r in refs)
    sim = client.post("/api/similarity", json={"text": HUMANISH + " " + AIISH}).json()
    assert sim["kind"] == "similarity" and sim["sources"]
    top = sim["sources"][0]
    assert top["title"] == "source essay" and top["exact_match_words"] > 50
    assert "independent of the AI-likelihood" in sim["note"]
    assert client.delete(f"/api/references/{add['id']}").status_code == 200


def test_settings_device(client):
    r = client.post("/api/settings", json={"device": "cpu"}).json()
    assert r["device"] == "cpu" and r["resolved_device"] == "cpu"
    assert client.post("/api/settings", json={"device": "tpu"}).status_code == 422
