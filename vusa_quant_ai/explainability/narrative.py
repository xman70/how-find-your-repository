"""Natural-language explanation layer.

The explanation is generated from the structured JSON produced by the quantitative
engine. Two modes:

* ``template`` (default, offline): deterministic text assembled from the JSON.
* ``llm`` (optional, needs ANTHROPIC_API_KEY): Claude rewrites the explanation, but it
  only receives the JSON and is instructed not to add numbers. A FABRICATION GUARD then
  extracts every number in the LLM output and checks it against the numbers present in
  the JSON (with rounding tolerance). Any untraceable number -> the LLM text is rejected
  and the deterministic template is used instead. The LLM can explain; it cannot invent.
"""
from __future__ import annotations

import json
import re

import numpy as np

from ..core.runtime import get_logger

log = get_logger("vusa.narrative")
MODEL = "claude-opus-5-5"

SYSTEM = """You are the explanation layer of a quantitative research application for the Vanguard S&P 500 UCITS ETF (VUSA).
You receive a JSON object produced by the quantitative engine. Write a clear, sober explanation for an investor.
Rules:
- Use ONLY numbers that appear in the JSON. Do not compute new numbers, do not round to new values beyond 1 decimal, do not add prices, dates, statistics, news or model results that are not in the JSON.
- Never claim certainty. This is decision support, not advice. Mention uncertainty and the counter-case.
- Structure: WHAT (signal and forecast), WHY (top evidence), HOW CONFIDENT, WHAT COULD INVALIDATE IT, WHAT HAPPENED HISTORICALLY, WHICH MODELS AGREE.
- Plain text, at most 350 words."""

_NUM = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:[.,]\d+)?%?")


def _collect_numbers(obj, out: set[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and np.isfinite(obj):
        out.add(float(obj))
        out.add(round(float(obj) * 100, 6))  # fractions shown as percentages
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            try:
                out.add(float(m.rstrip("%").replace(",", "")))
            except ValueError:
                pass


def fabrication_check(text: str, payload: dict, tol: float = 0.051) -> list[str]:
    """Return the numbers in ``text`` that cannot be traced to ``payload``."""
    allowed: set[float] = set(range(0, 11)) | {20.0, 50.0, 80.0, 90.0, 95.0, 99.0, 100.0}
    _collect_numbers(payload, allowed)
    allowed_arr = np.array(sorted(allowed))
    bad = []
    for m in _NUM.findall(text):
        try:
            v = float(m.rstrip("%").replace(",", ""))
        except ValueError:
            continue
        if not len(allowed_arr) or np.min(np.abs(allowed_arr - v)) > max(tol, abs(v) * 0.006):
            bad.append(m)
    return bad


def template_explanation(p: dict) -> str:
    d, fc = p.get("decision", {}), p.get("forecast_primary", {})
    lines = []
    lines.append(f"WHAT: {d.get('label')} - opportunity {d.get('opportunity')}/100, risk {d.get('risk')}/100, "
                 f"confidence {100 * (d.get('confidence') or 0):.0f}%.")
    if fc:
        lo, hi = fc.get("interval", [None, None])
        lines.append(f"The {fc.get('horizon')}D ensemble expects {100 * fc.get('expected_return', 0):+.1f}% with an "
                     f"{100 * fc.get('interval_level', 0.8):.0f}% interval of {100 * lo:+.1f}% to {100 * hi:+.1f}%; "
                     f"calibrated P(up) {100 * fc.get('p_up', 0):.0f}%.")
    contrib = sorted(d.get("contributions", {}).items(), key=lambda kv: -abs(kv[1]))
    lines.append("WHY: " + "; ".join(f"{k} {v:+.1f} pts" for k, v in contrib[:6]) + ".")
    lines.append(f"HOW CONFIDENT: component agreement {100 * (d.get('agreement') or 0):.0f}%; penalties: " +
                 (", ".join(f"{k} {100 * v:.0f}%" for k, v in d.get("penalties", {}).items() if v) or "none") + ".")
    wwc = d.get("what_would_change", [])
    if wwc:
        lines.append("WHAT COULD INVALIDATE IT: " + " | ".join(wwc[:3]))
    an = p.get("analogues", {}).get("summary", {})
    if an:
        lines.append(f"HISTORICALLY: closest analogues averaged {100 * an['ret_20d']['weighted_mean']:+.1f}% over 20D "
                     f"({100 * (an['ret_20d']['p_positive'] or 0):.0f}% positive).")
    dis = fc.get("disagreement", {}) if fc else {}
    if dis.get("sign_agreement") is not None:
        lines.append(f"MODELS: {100 * dis['sign_agreement']:.0f}% of {dis.get('n_models')} models agree on direction.")
    for c in d.get("contradictions", [])[:2]:
        lines.append(f"NOTE: {c}")
    lines.append("This is a probabilistic research assessment, not a guarantee or investment advice.")
    return "\n".join(lines)


def llm_explanation(payload: dict, api_key: str, timeout: float = 60.0) -> tuple[str, dict]:
    """Return (text, meta). Falls back to the template on any failure or guard violation."""
    meta = {"mode": "template", "model": None, "rejected_numbers": []}
    if not api_key:
        meta["reason"] = "no ANTHROPIC_API_KEY configured"
        return template_explanation(payload), meta
    try:
        import anthropic
    except ImportError:
        meta["reason"] = "anthropic package not installed"
        return template_explanation(payload), meta
    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        body = json.dumps(payload, default=str)[:60000]
        kwargs = dict(model=MODEL, max_tokens=4000, system=SYSTEM, output_config={"effort": "medium"},
                      messages=[{"role": "user", "content": f"Quantitative engine output (JSON):\n{body}"}])
        try:  # server-side refusal fallback (opt-in)
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        except TypeError:  # older SDK without the typed parameter
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                               extra_body={"fallbacks": "default"}, **kwargs)
        if resp.stop_reason == "refusal":
            meta["reason"] = "model declined"
            return template_explanation(payload), meta
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        bad = fabrication_check(text, payload)
        if bad:
            meta.update({"reason": "fabrication guard rejected LLM text", "rejected_numbers": bad[:20]})
            log.warning("LLM explanation rejected: untraceable numbers %s", bad[:10])
            return template_explanation(payload), meta
        meta.update({"mode": "llm", "model": resp.model})
        return text, meta
    except Exception as exc:  # noqa: BLE001
        meta["reason"] = f"LLM call failed: {type(exc).__name__}: {exc}"[:300]
        return template_explanation(payload), meta
