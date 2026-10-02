import json

from aidetect.report import to_html, to_json, to_pdf, write_report
from aidetect.visualization import diverging, highlight_css, report_charts


def test_diverging_scale_endpoints():
    assert diverging(0.0) == "#2a78d6"
    assert diverging(0.5) == "#f0efec"
    assert diverging(1.0) == "#e34948"
    assert highlight_css(0.5).endswith(",0.000)")


def test_charts_render(ok_result):
    charts = report_charts(ok_result, "svg")
    for key in ("sentence_probability", "perplexity", "burstiness", "vocabulary", "sentence_lengths", "heatmap",
                "disagreement"):
        assert charts[key].lstrip().startswith(("<?xml", "<svg"))
    png = report_charts(ok_result, "png")["heatmap"]
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_html_report_contains_required_sections_and_escapes(ok_result):
    res = json.loads(json.dumps(ok_result))
    res["sentences"][0]["text"] = "<script>alert(1)</script> injected"
    html = to_html(res)
    for needle in ("Document analysis report", "Model disagreement", "Heatmap", "Statistical analysis",
                   "Sentence-by-sentence", "Limitations", "Disclaimer", "Potential authorship/style transitions"):
        assert needle in html
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_pdf_and_json(ok_result, tmp_path):
    pdf = to_pdf(ok_result)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 10_000
    data = json.loads(to_json(ok_result))
    assert data["result"]["document_probability"] == ok_result["result"]["document_probability"]
    for ext in ("json", "html", "pdf"):
        p = write_report(ok_result, tmp_path / f"r.{ext}")
        assert p.stat().st_size > 0


def test_error_results_still_produce_reports():
    bad = {"status": "insufficient_text", "message": "too short", "meta": {}, "disclaimer": "d"}
    assert "too short" in to_html(bad)
    assert to_pdf(bad)[:5] == b"%PDF-"
