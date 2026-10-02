"""Report generation: JSON, self-contained HTML, and PDF."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import DISCLAIMER
from .visualization import diverging, highlight_css, report_charts

TEMPLATES = Path(__file__).parent / "templates"


def _pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{100 * v:.0f}%"


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def to_json(result: dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False, default=lambda o: None)


def _paragraph_groups(result: dict) -> list[list[dict]]:
    groups: dict[int, list] = {}
    for s in result["sentences"]:
        groups.setdefault(s["paragraph"], []).append(s)
    return [groups[k] for k in sorted(groups)]


def to_html(result: dict) -> str:
    if result.get("status") != "ok":
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>Document Analysis Report</title></head>"
                f"<body style='font-family:system-ui;background:#f9f9f7;max-width:760px;margin:40px auto;padding:0 16px'>"
                f"<h1>Document analysis report</h1><p><strong>{result.get('status')}</strong>: "
                f"{result.get('message', '')}</p><p>{DISCLAIMER}</p></body></html>")
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html", "j2"]))
    tpl = env.get_template("report.html.j2")
    charts = report_charts(result, "svg")
    return tpl.render(r=result, res=result["result"], charts=charts, paragraphs=_paragraph_groups(result),
                      pct=_pct, fmt=_fmt, hl=highlight_css)


def to_pdf(result: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="Document Analysis Report")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12, alignment=TA_LEFT)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor("#52514e"))
    h1, h2 = ss["Heading1"], ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=10)
    notice = ParagraphStyle("n", parent=body, backColor=colors.HexColor("#fff7e6"), borderPadding=6,
                            borderColor=colors.HexColor("#eda100"), borderWidth=0.5, spaceBefore=6, spaceAfter=8)
    E = escape
    story = [Paragraph("Document analysis report", h1)]
    m = result.get("meta", {})
    if result.get("status") != "ok":
        story += [Paragraph(E(f"{result.get('status')}: {result.get('message', '')}"), body),
                  Paragraph(E(DISCLAIMER), notice)]
        doc.build(story)
        return buf.getvalue()
    r = result["result"]
    tstyle = TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e1e0d9")),
                         ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#52514e")), ("VALIGN", (0, 0), (-1, -1), "TOP")])
    story += [
        Paragraph(E("DOCUMENT ANALYSIS"), h2),
        Table([["Document", m.get("title")], ["Date", m.get("date")], ["Word count", m.get("word_count")],
               ["Language", m.get("language")], ["Model", result["model"]["version"]],
               ["Privacy", result.get("privacy")]], colWidths=[35 * mm, 140 * mm], style=tstyle),
        Paragraph(E(result["disclaimer"]), notice),
        Paragraph("RESULT", h2),
        Table([["AI-associated probability (share of text)", _pct(r["ai_associated_share"])],
               ["Human-associated probability (share of text)", _pct(r["human_associated_share"])],
               ["Uncertain / mixed", _pct(r["uncertain_share"])],
               ["Document-level AI probability (calibrated)", _pct(r["document_probability"])],
               ["Category", r["hybrid"]["label"]]], colWidths=[95 * mm, 80 * mm], style=tstyle),
        Paragraph(E(r["interpretation"]), body),
        Paragraph(E(r["share_definition"]), small),
        Paragraph("CONFIDENCE", h2),
        Paragraph(E(f"{r['confidence_level']} ({_pct(r['confidence'])}) - {r['evidence_text']}. Confidence is "
                    "separate from likelihood: it reflects text length, detector agreement, feature coverage and "
                    "committee stability."), body),
        Table([[k.replace("_", " "), f"{v:.2f}"] for k, v in r["confidence_components"].items()],
              colWidths=[95 * mm, 80 * mm], style=tstyle),
    ]
    for w in result.get("warnings", []):
        story.append(Paragraph(E(w), notice))
    charts = report_charts(result, "png")

    def img(key, width=175 * mm):
        if not charts.get(key):
            return Spacer(1, 1)
        from reportlab.lib.utils import ImageReader

        ir = ImageReader(io.BytesIO(charts[key]))
        w, h = ir.getSize()
        return Image(io.BytesIO(charts[key]), width=width, height=width * h / w)

    story += [Paragraph("MODEL DISAGREEMENT", h2), Paragraph(E(result["disagreement"]["note"]), body), img("disagreement"),
              Table([["Detector", "AI prob.", "± error", "weight"]] +
                    [[Paragraph(E(d["label"]), small), _pct(d["probability"]), f"{d['error_estimate']:.3f}",
                      f"{d['weight']:.2f}"] for d in result["detectors"]],
                    colWidths=[105 * mm, 25 * mm, 22 * mm, 22 * mm], style=tstyle),
              Paragraph("SIGNALS", h2)]
    for line in result["explanation"]["summary"].split("\n"):
        story.append(Paragraph(E(line), body))
    story += [Paragraph("STATISTICAL ANALYSIS", h2)]
    for group, g in result["statistics"].items():
        rows = [["Measure", "Value", "Human pct.", "Human median", "AI median"]]
        for it in g["items"]:
            rows.append([Paragraph(E(it["label"]), small), it["display"], _fmt(it["human_percentile"]),
                         _fmt(it["human_median"]), _fmt(it["ai_median"])])
        story += [Paragraph(E(group.capitalize()), ParagraphStyle("h3", parent=body, fontName="Helvetica-Bold")),
                  Table(rows, colWidths=[70 * mm, 25 * mm, 25 * mm, 27 * mm, 27 * mm], style=tstyle)]
    story += [PageBreak(), Paragraph("HEATMAP", h2), img("heatmap")]
    for para in _paragraph_groups(result):
        parts = []
        for s in para:
            col = diverging(s["probability"])
            parts.append(f'<font size="6" color="#898781">{s["number"]}</font> '
                         f'<span backColor="{_tint(col, s["probability"])}">{E(s["text"])}</span>')
        story.append(Paragraph(" ".join(parts), body))
        story.append(Spacer(1, 4))
    story += [Paragraph("GRAPHS", h2)]
    for key in ("sentence_probability", "perplexity", "burstiness", "style_drift", "vocabulary", "sentence_lengths"):
        story.append(img(key))
    story += [Paragraph("POTENTIAL AUTHORSHIP/STYLE TRANSITIONS", h2)]
    cps = result["transitions"]["changepoints"]
    if cps:
        rows = [["Before sentence", "p-value", "Style JS", "Perplexity Δ", "Semantic dist.", "AI before→after"]]
        for c in cps:
            d = c["differences"]
            rows.append([c["boundary_before_sentence"] + 1, f"{c['p_value']:.3f}", _fmt(d["style_js"]),
                         _fmt(d["perplexity_bits"]), _fmt(d["semantic_distance"]),
                         f"{_pct(c['ai_probability_before'])} → {_pct(c['ai_probability_after'])}"])
        story += [Table(rows, style=tstyle),
                  Paragraph("A transition marks a statistically unusual change in writing characteristics; it does not "
                            "mean that AI use begins or ends there.", small)]
    else:
        story.append(Paragraph("No statistically significant style transition was detected.", body))
    story += [Paragraph("SECTION ANALYSIS (sentence by sentence)", h2)]
    rows = [["#", "Sentence", "AI lik.", "Conf."]]
    for s in result["sentences"]:
        rows.append([s["number"], Paragraph(E(s["text"]), small), _pct(s["probability"]), s["confidence"]])
    story.append(Table(rows, colWidths=[9 * mm, 136 * mm, 16 * mm, 15 * mm], style=tstyle, repeatRows=1))
    story += [Paragraph("LIMITATIONS", h2)] + [Paragraph(E("• " + x), body) for x in result["limitations"]]
    story.append(Paragraph(E(result["disclaimer"]), notice))
    doc.build(story)
    return buf.getvalue()


def _tint(hex_color: str, p: float) -> str:
    a = min(0.55, abs(p - 0.5) * 2 * 0.55)
    c = hex_color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(255 + (v - 255) * a) for v in rgb]
    return "#" + "".join(f"{v:02x}" for v in mixed)


def write_report(result: dict, path: str | Path) -> Path:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".json":
        path.write_text(to_json(result), encoding="utf-8")
    elif ext in (".html", ".htm"):
        path.write_text(to_html(result), encoding="utf-8")
    elif ext == ".pdf":
        path.write_bytes(to_pdf(result))
    else:
        raise ValueError("Report format must be .json, .html or .pdf")
    return path
