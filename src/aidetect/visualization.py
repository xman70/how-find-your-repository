"""Static charts (matplotlib) for HTML/PDF reports and the evaluation report.

Colour roles follow one documented palette:
* AI likelihood uses a diverging scale - blue (human-associated) <-> neutral gray
  (uncertain, 0.5) <-> red (AI-associated) - so "uncertain" carries no colour.
* Single-series charts use categorical slot 1 (blue); multi-series charts use
  slots 1..3 in fixed order with a legend and direct labels.
* Status colours are never used for data. No chart has two y-axes.
The interactive web UI draws the same charts with Plotly (web/app.js).
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DIV_LOW, DIV_MID, DIV_HIGH = "#2a78d6", "#f0efec", "#e34948"


def _hex(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def diverging(p: float) -> str:
    """Blue (0) -> gray (0.5) -> red (1)."""
    p = float(np.clip(p, 0, 1))
    if p < 0.5:
        a, b, t = _hex(DIV_LOW), _hex(DIV_MID), p / 0.5
    else:
        a, b, t = _hex(DIV_MID), _hex(DIV_HIGH), (p - 0.5) / 0.5
    rgb = a + (b - a) * t
    return "#" + "".join(f"{int(round(v)):02x}" for v in rgb)


def highlight_css(p: float, dark: bool = False) -> str:
    """Translucent background for highlighted text (intensity grows away from 0.5)."""
    a = min(0.55, abs(p - 0.5) * 2 * 0.55)
    if p >= 0.5:
        rgb = "230,103,103" if dark else "227,73,72"
    else:
        rgb = "57,135,229" if dark else "42,120,214"
    return f"rgba({rgb},{a:.3f})"


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def _out(fig, fmt: str):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150 if fmt == "png" else None, facecolor=SURFACE)
    plt.close(fig)
    data = buf.getvalue()
    return data.decode("utf-8") if fmt == "svg" else data


def _line(values, title, ylabel, fmt, ref=None, ref_label=None, x=None, marker_colors=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    v = np.array([np.nan if a is None else a for a in values], dtype=float)
    xs = np.arange(1, len(v) + 1) if x is None else np.asarray(x, float) + 1
    _style(ax, title, "Sentence", ylabel)
    color = INK2 if marker_colors is not None else SERIES[0]
    ax.plot(xs, v, color=color, linewidth=1.5)
    if marker_colors is not None:
        ax.scatter(xs, v, c=marker_colors, s=22, zorder=3, edgecolors=SURFACE, linewidths=1.2)
    for r, lab in zip(ref or [], ref_label or []):
        ax.axhline(r, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(xs[-1] if len(xs) else 1, r, f" {lab}", va="center", fontsize=7, color=INK2)
    if ylim:
        ax.set_ylim(*ylim)
    return _out(fig, fmt)


def report_charts(res: dict, fmt: str = "svg") -> dict[str, object]:
    s = res["series"]
    sp = s["sentence_probability"]
    b = res["result"]["bands"]
    charts = {
        "sentence_probability": _line(sp, "Sentence AI likelihood (calibrated)", "Probability", fmt,
                                      ref=[b["t_high"], b["t_low"]], ref_label=["AI-associated band", "human-associated band"],
                                      marker_colors=[diverging(p) for p in sp], ylim=(0, 1)),
        "perplexity": _line(s["sentence_surprisal"], "Sentence perplexity (mean surprisal)",
                            "bits/word" if res["model"]["predictability_backend"] == "unigram" else "bits/token", fmt),
        "burstiness": _line(s["rolling_perplexity_std"], "Burstiness: local variation of predictability", "std (bits)", fmt),
        "vocabulary": _line(s["rolling_mattr"], "Vocabulary diversity (MATTR, 5-sentence window)", "MATTR", fmt),
    }
    drift = s["style_drift"]
    if drift["values"]:
        charts["style_drift"] = _line(drift["values"], "Style drift between consecutive windows", "JS divergence", fmt,
                                      x=np.array(drift["centers"]))
    # sentence-length distribution
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    _style(ax, "Sentence-length distribution", "Words per sentence", "Sentences")
    lens = np.array(s["sentence_lengths"])
    bins = np.arange(0, max(lens.max() + 5, 10), 5)
    ax.hist(lens, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    charts["sentence_lengths"] = _out(fig, fmt)
    # heatmap strip
    n = len(sp)
    cols = min(25, n)
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * rows + 0.6))
    _style(ax, "AI-likelihood heatmap (one cell per sentence)")
    ax.grid(False)
    for i, p in enumerate(sp):
        r, c = divmod(i, cols)
        ax.add_patch(plt.Rectangle((c, -r), 0.92, 0.85, color=diverging(p), linewidth=0))
        ax.text(c + 0.46, -r + 0.42, str(i + 1), ha="center", va="center", fontsize=5.5,
                color=INK if 0.25 < p < 0.75 else SURFACE)
    ax.set_xlim(-0.1, cols)
    ax.set_ylim(-rows + 0.8, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    charts["heatmap"] = _out(fig, fmt)
    # model disagreement
    dets = res["detectors"]
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(dets) + 0.9))
    _style(ax, "Model disagreement: individual detector outputs", "Calibrated AI probability")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    labels = [d["label"].split(" (")[0] for d in dets][::-1]
    vals = [d["probability"] for d in dets][::-1]
    errs = [d["error_estimate"] for d in dets][::-1]
    y = np.arange(len(dets))
    ax.barh(y, vals, height=0.55, color=SERIES[0], xerr=errs, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ens = res["result"]["document_probability"]
    ax.axvline(ens, color=INK, linewidth=1.0, linestyle=(0, (4, 2)))
    ax.text(ens, len(dets) - 0.45, f" ensemble {ens:.0%}", fontsize=7, color=INK, va="bottom")
    for yy, v in zip(y, vals):
        ax.text(1.02, yy, f"{v:.0%}", va="center", fontsize=7, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5, color=INK2)
    ax.set_xlim(0, 1)
    charts["disagreement"] = _out(fig, fmt)
    return charts


# --------------------------------------------------------------------------- evaluation figures
def reliability_figure(rel_by_method: dict[str, dict], title: str, fmt: str = "png"):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    _style(ax, title, "Mean predicted probability", "Observed frequency of AI text")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    names = {"none": "uncalibrated", "platt": "Platt", "isotonic": "isotonic"}
    for k, (m, rel) in enumerate(rel_by_method.items()):
        pts = [(b["mean_pred"], b["frac_pos"]) for b in rel["bins"] if b["count"]]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=SERIES[k % 3], linewidth=1.5, marker="o", markersize=4, label=names.get(m, m))
        ax.text(x[-1], y[-1], f" {names.get(m, m)}", fontsize=7, color=INK2, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    return _out(fig, fmt)


def length_figure(rows: list[dict], thresholds: dict, fmt: str = "png"):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    _style(ax, "Discrimination vs. text length (held-out test documents)", "Words", "ROC-AUC")
    x = [r["words"] for r in rows]
    y = [r["roc_auc"] for r in rows]
    ax.plot(x, y, color=SERIES[0], linewidth=1.5, marker="o", markersize=4)
    for name, v in thresholds.items():
        if v:
            ax.axvline(v, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
            ax.text(v, min(y) if y else 0.5, f" {name}", fontsize=7, color=INK2, rotation=90, va="bottom")
    ax.set_ylim(0.5, 1.0)
    return _out(fig, fmt)


def grouped_bars(categories: list[str], series: dict[str, list[float]], title: str, ylabel: str, fmt: str = "png",
                 ylim=(0, 1), ref: float | None = None, ref_label: str | None = None, errors: dict | None = None):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    _style(ax, title, "", ylabel)
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(categories))
    for k, (name, vals) in enumerate(series.items()):
        err = None
        if errors and name in errors:
            e = np.array(errors[name], dtype=float)
            err = np.vstack([np.maximum(0, np.array(vals) - e[:, 0]), np.maximum(0, e[:, 1] - np.array(vals))])
        ax.bar(x + (k - (n - 1) / 2) * width, vals, width=width * 0.92, color=SERIES[k % 3], label=name,
               yerr=err, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    if ref is not None:
        ax.axhline(ref, color=INK, linewidth=0.9, linestyle=(0, (4, 2)))
        if ref_label:
            ax.text(len(categories) - 0.5, ref, f" {ref_label}", fontsize=7, color=INK, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7, color=INK2, rotation=30, ha="right")
    ax.set_ylim(*ylim)
    if n > 1:
        ax.legend(frameon=False, fontsize=7, ncol=n, loc="upper left")
    return _out(fig, fmt)
