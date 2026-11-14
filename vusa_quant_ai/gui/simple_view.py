"""Simple overview: two traffic lights (short term / long term) in plain language."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

COLORS = {"BUY": "#2ecc71", "HOLD": "#f1c40f", "SELL": "#e74c3c", None: "#3a4350"}
MEANING = {"BUY": "Conditions look favourable", "HOLD": "No clear edge - wait / keep position",
           "SELL": "Conditions look unfavourable"}


def _pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100 * x:{'+' if sign else ''}.{d}f}%"


def _conf_word(c: float) -> str:
    return "very low" if c < 0.15 else "low" if c < 0.35 else "moderate" if c < 0.6 else "high"


class LightPanel(QFrame):
    def __init__(self, title: str, subtitle: str):
        super().__init__()
        self.setObjectName("kpi")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 16, 22, 16)
        lay.setSpacing(8)
        head = QLabel(f"<span style='font-size:17pt;font-weight:bold'>{title}</span>"
                      f"<br><span style='color:#8b949e;font-size:11pt'>{subtitle}</span>")
        head.setAlignment(Qt.AlignCenter)
        lay.addWidget(head)
        self.light = QLabel("")
        self.light.setFixedSize(170, 170)
        lay.addWidget(self.light, alignment=Qt.AlignCenter)
        self.word = QLabel("—")
        self.word.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.word)
        self.meaning = QLabel("Click ANALYZE VUSA NOW")
        self.meaning.setAlignment(Qt.AlignCenter)
        self.meaning.setWordWrap(True)
        lay.addWidget(self.meaning)
        self.conf_lbl = QLabel("Confidence")
        self.conf = QProgressBar()
        self.conf.setMaximum(100)
        self.conf.setTextVisible(False)
        self.conf.setFixedHeight(12)
        lay.addWidget(self.conf_lbl)
        lay.addWidget(self.conf)
        self.numbers = QLabel("")
        self.numbers.setWordWrap(True)
        self.numbers.setTextFormat(Qt.RichText)
        lay.addWidget(self.numbers)
        self.reasons = QLabel("")
        self.reasons.setWordWrap(True)
        self.reasons.setTextFormat(Qt.RichText)
        lay.addWidget(self.reasons)
        self.track = QLabel("")
        self.track.setWordWrap(True)
        self.track.setTextFormat(Qt.RichText)
        self.track.setStyleSheet("color:#aab4c0")
        lay.addWidget(self.track)
        lay.addStretch(1)
        self._paint(None)

    def _paint(self, signal):
        col = COLORS[signal]
        self.light.setStyleSheet(f"background:{col}; border-radius:85px; border:6px solid #1c232c;")

    def set(self, t: dict, currency: str = "€"):
        sig = t["signal"]
        self._paint(sig)
        self.word.setText(f"<span style='font-size:30pt;font-weight:bold;color:{COLORS[sig]}'>{sig}</span>")
        extra = ""
        if "LOW CONFIDENCE" in t.get("strength", ""):
            raw = t["strength"].split("raw ")[-1]
            extra = f" The evidence leans {raw}, but confidence is too low to show it as a signal."
        elif t.get("strength") == "STRONG":
            extra = " Several independent indicators agree (strong signal)."
        elif "WEAK" in t.get("strength", ""):
            extra = f" Leaning {t['strength'].split()[-1].lower()}, but not enough for a signal."
        self.meaning.setText(MEANING[sig] + "." + extra)
        c = t.get("confidence") or 0.0
        self.conf.setValue(int(100 * c))
        self.conf_lbl.setText(f"Confidence: {100 * c:.0f}% ({_conf_word(c)})")
        hl = t.get("headline")
        biased = bool(hl) and any(w.startswith("Residual bias") for w in hl.get("warnings", []))
        if hl and biased:
            self.numbers.setText(
                f"<b>Next {hl['horizon']} trading days:</b> the model's forecasts for this period have been "
                f"systematically off in the past, so no reliable expected figure can be shown.<br>"
                f"<b>Chance of a rise:</b> {_pct(hl['p_up'], 0, False)} "
                f"(normal for this period: {_pct(hl['base_rate'], 0, False)}).")
        elif hl:
            lo, hi = hl["interval"]
            self.numbers.setText(
                f"<b>Next {hl['horizon']} trading days:</b> expected {_pct(hl['expected_return'])}; "
                f"{int(100 * hl.get('interval_level', 0.8))}% of outcomes between {_pct(lo)} and {_pct(hi)}.<br>"
                f"<b>Chance of a rise:</b> {_pct(hl['p_up'], 0, False)} "
                f"(normal for this period: {_pct(hl['base_rate'], 0, False)}).")
        else:
            self.numbers.setText("No forecast available for this time frame in the selected mode.")
        f_ = "".join(f"<div style='color:#2ecc71'>+ {x}</div>" for x in t.get("reasons_for", [])) or "<div>+ nothing notable</div>"
        a_ = "".join(f"<div style='color:#e74c3c'>− {x}</div>" for x in t.get("reasons_against", [])) or "<div>− nothing notable</div>"
        self.reasons.setText(f"<b>Why:</b>{f_}{a_}")
        tr = t.get("track_record") or {}
        if tr.get("available"):
            rows = []
            for col, name in (("GREEN", "green"), ("YELLOW", "yellow"), ("RED", "red")):
                b = tr["by_light"].get(col, {})
                if b.get("n"):
                    rows.append(f"after {name}: up {_pct(b['p_up'], 0, False)} of the time, average {_pct(b['mean_return'])}")
                else:
                    rows.append(f"{name}: never shown")
            self.track.setText(f"<b>Past record</b> ({tr['horizon']}-day results, {tr['from'][:4]}–{tr['to'][:4]}; "
                               f"all days: up {_pct(tr['base_p_up'], 0, False)}, average {_pct(tr['base_mean'])}):<br>"
                               + "<br>".join(rows) + f"<br><i>{tr['verdict']}</i>")
        else:
            self.track.setText(f"<b>Past record:</b> not available ({tr.get('reason', 'n/a')}).")


class SimpleView(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self.top = QLabel("<span style='font-size:13pt'>No analysis yet. Click <b>ANALYZE VUSA NOW</b>.</span>")
        self.top.setAlignment(Qt.AlignCenter)
        self.top.setTextFormat(Qt.RichText)
        lay.addWidget(self.top)
        row = QHBoxLayout()
        from ..signals.terms import TERMS

        self.panels = {k: LightPanel(v["title"], v["subtitle"]) for k, v in TERMS.items()}
        for p in self.panels.values():
            row.addWidget(p)
        lay.addLayout(row, 1)
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(Qt.RichText)
        lay.addWidget(self.notes)
        foot = QLabel("<span style='color:#8b949e'><b style='color:#2ecc71'>GREEN = BUY</b>: conditions favourable for "
                      "buying / adding. &nbsp; <b style='color:#f1c40f'>YELLOW = HOLD</b>: no clear edge, wait. &nbsp; "
                      "<b style='color:#e74c3c'>RED = SELL</b>: conditions unfavourable. &nbsp; The lights are probabilities, "
                      "not certainties: this is research support, not investment advice, and it never trades. "
                      "Tick <b>Advanced view</b> (top right) for all details.</span>")
        foot.setWordWrap(True)
        foot.setTextFormat(Qt.RichText)
        lay.addWidget(foot)

    def populate(self, st: dict):
        cur = "€" if st.get("currency") == "EUR" else ""
        self.top.setText(f"<span style='font-size:13pt'>VUSA <b>{cur}{st['price']:.2f}</b> &nbsp;·&nbsp; close of "
                         f"{st['as_of']} &nbsp;·&nbsp; market mood: <b>{st['regime']['regime']}</b>, "
                         f"{st['regime']['volatility_label'].lower()} volatility</span>")
        for k, p in self.panels.items():
            if k in st.get("terms", {}):
                p.set(st["terms"][k], cur)
        notes = []
        s, l_ = st.get("terms", {}).get("short"), st.get("terms", {}).get("long")
        if s and l_ and s["signal"] != l_["signal"]:
            notes.append(f"The two time frames disagree (short term {s['signal']}, long term {l_['signal']}). That is normal: "
                         "choose the light that matches how long you plan to hold.")
        for t in (s, l_):
            if t:
                notes += [f"{t['title']}: {n}" for n in t.get("notes", []) if "suppressed" not in n]
                for w in (t.get("headline") or {}).get("warnings", []):
                    if w.startswith("Only ~"):
                        notes.append(f"{t['title']}: based on very few independent past periods - the numbers are rough.")
                    elif w.startswith("Residual bias"):
                        notes.append(f"{t['title']}: past forecasts for this period were systematically off - low reliability.")
        self.notes.setText("".join(f"<div style='color:#f1c40f'>• {n}</div>" for n in notes[:6]))
