"""Reusable Qt widgets: Plotly chart view (QtWebEngine with browser fallback), pandas
table, KPI cards, background worker."""
from __future__ import annotations

import tempfile
import traceback
import uuid
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSizePolicy, QTableView,
                               QTextBrowser, QVBoxLayout, QWidget)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView

    HAS_WEBENGINE = True
except Exception:  # pragma: no cover - depends on system libraries
    HAS_WEBENGINE = False

CHART_DIR = Path(tempfile.gettempdir()) / "vusa_quant_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


class ChartView(QWidget):
    """Displays a Plotly figure. Uses a local plotly.min.js (offline) via a temp HTML file."""

    def __init__(self, parent=None, min_height: int = 380):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._path: Path | None = None
        if HAS_WEBENGINE:
            self.view = QWebEngineView(self)
            self.view.setMinimumHeight(min_height)
            lay.addWidget(self.view)
        else:
            self.view = QTextBrowser(self)
            self.view.setText("QtWebEngine unavailable - use 'Open in browser'.")
            lay.addWidget(self.view)
        btn = QPushButton("Open in browser")
        btn.setMinimumWidth(150)
        btn.setObjectName("secondary")
        btn.clicked.connect(self.open_external)
        lay.addWidget(btn, alignment=Qt.AlignRight)

    def set_figure(self, fig) -> None:
        self._path = CHART_DIR / f"chart_{uuid.uuid4().hex[:10]}.html"
        fig.write_html(str(self._path), include_plotlyjs="directory", full_html=True,
                       config={"displaylogo": False, "responsive": True})
        if HAS_WEBENGINE:
            self.view.setUrl(QUrl.fromLocalFile(str(self._path)))

    def open_external(self):
        if self._path:
            webbrowser.open(self._path.as_uri())


class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None):
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()

    def set(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.reset_index(drop=True) if df is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        v = self._df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if isinstance(v, (float, np.floating)):
                if not np.isfinite(v):
                    return "n/a"
                return f"{v:.4f}" if abs(v) < 1000 else f"{v:,.0f}"
            if isinstance(v, (list, dict)):
                return str(v)[:300]
            return "" if v is None else str(v)
        if role == Qt.ForegroundRole and isinstance(v, str):
            u = v.upper()
            if "BUY" in u or u in ("PASS", "OK", "BULLISH", "STABLE"):
                return QColor("#2ecc71")
            if "SELL" in u or u in ("FAIL", "FAILED", "BEARISH", "UNSTABLE"):
                return QColor("#e74c3c")
            if "HOLD" in u or u in ("WARN", "CACHED", "MODERATELY STABLE"):
                return QColor("#f1c40f")
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)


class DataTable(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_ = PandasModel()
        self.setModel(self.model_)
        self.setSortingEnabled(False)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.setWordWrap(True)

    def set_df(self, df: pd.DataFrame | None, max_rows: int = 3000):
        self.model_.set(df.head(max_rows) if df is not None else None)
        self.resizeColumnsToContents()
        for c in range(self.model_.columnCount()):
            if self.columnWidth(c) > 420:
                self.setColumnWidth(c, 420)


class KPICard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("kpi")
        self.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        self.t = QLabel(title.upper())
        self.t.setObjectName("kpiTitle")
        self.v = QLabel("—")
        self.v.setObjectName("kpiValue")
        f = QFont()
        f.setPointSize(17)
        f.setBold(True)
        self.v.setFont(f)
        self.sub = QLabel("")
        self.sub.setObjectName("kpiSub")
        self.sub.setWordWrap(True)
        for w in (self.t, self.v, self.sub):
            lay.addWidget(w)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set(self, value: str, sub: str = "", color: str | None = None):
        self.v.setText(value)
        self.sub.setText(sub)
        self.v.setStyleSheet(f"color:{color};" if color else "")


def kpi_row(titles: list[str]) -> tuple[QWidget, dict[str, KPICard]]:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    cards = {}
    for t in titles:
        cards[t] = KPICard(t)
        lay.addWidget(cards[t])
    return w, cards


class Worker(QObject):
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            res = self.fn(*self.args, progress=lambda m, f: self.progress.emit(m, float(f)), **self.kwargs)
            self.finished.emit(res)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()[-2500:]}")


def start_worker(owner, fn, on_done, on_fail, on_progress, *args, **kwargs) -> tuple[QThread, Worker]:
    th = QThread(owner)
    wk = Worker(fn, *args, **kwargs)
    wk.moveToThread(th)
    th.started.connect(wk.run)
    wk.progress.connect(on_progress)
    wk.finished.connect(on_done)
    wk.failed.connect(on_fail)
    wk.finished.connect(th.quit)
    wk.failed.connect(th.quit)
    th.start()
    return th, wk


DARK_QSS = """
QWidget { background:#0f1419; color:#d8dee9; font-family:'Segoe UI', Arial; font-size:10pt; }
QTabWidget::pane { border:1px solid #2b3540; }
QTabBar::tab { background:#17202a; padding:6px 10px; border:1px solid #2b3540; border-bottom:none; }
QTabBar::tab:selected { background:#1f2d3a; color:#ffffff; }
QPushButton { background:#1f6feb; color:white; border:none; padding:7px 14px; border-radius:4px; font-weight:bold; }
QPushButton:hover { background:#388bfd; } QPushButton:disabled { background:#30363d; color:#8b949e; }
QPushButton#deep { background:#8957e5; } QPushButton#secondary { background:#30363d; }
QFrame#kpi { background:#161b22; border:1px solid #30363d; border-radius:6px; }
QLabel#kpiTitle { color:#8b949e; font-size:8pt; } QLabel#kpiSub { color:#8b949e; font-size:8pt; }
QLabel#banner { padding:6px; border-radius:4px; font-weight:bold; }
QTableView { background:#0d1117; alternate-background-color:#161b22; gridline-color:#30363d; }
QHeaderView::section { background:#161b22; color:#c9d1d9; padding:4px; border:1px solid #30363d; }
QTextBrowser, QPlainTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#0d1117; border:1px solid #30363d; }
QProgressBar { border:1px solid #30363d; text-align:center; } QProgressBar::chunk { background:#1f6feb; }
QGroupBox { border:1px solid #30363d; margin-top:12px; padding-top:8px; } QGroupBox::title { color:#58a6ff; }
"""
