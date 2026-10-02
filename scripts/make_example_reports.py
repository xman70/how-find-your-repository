"""Produce example analyses (reports/examples/) from the committed example dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidetect.data import load_jsonl  # noqa: E402
from aidetect.predict import Analyzer  # noqa: E402
from aidetect.report import write_report  # noqa: E402


def main() -> None:
    out = ROOT / "reports" / "examples"
    out.mkdir(parents=True, exist_ok=True)
    ex = load_jsonl(ROOT / "data" / "example" / "example_dataset.jsonl")
    picks = {
        "human_essay": next(s for s in ex if s["author_type"] == "human" and s["domain"] == "essay" and s["n_words"] > 300),
        "ai_essay": next(s for s in ex if s["author_type"] == "ai" and s["domain"] == "essay" and s["n_words"] > 300),
        "hybrid_news": next(s for s in ex if s["author_type"] == "hybrid_spliced" and s["n_words"] > 250),
    }
    an = Analyzer(device="cpu")
    summary = []
    for name, s in picks.items():
        res = an.analyze(s["text"], title=f"Example: {name.replace('_', ' ')} (example dataset, CC BY 3.0)")
        for ext in ("html", "pdf", "json"):
            write_report(res, out / f"{name}.{ext}")
        r = res["result"]
        summary.append({"example": name, "ground_truth": {"author_type": s["author_type"], "generator": s["generator"],
                                                          "ai_fraction": s["ai_fraction"]},
                        "document_probability": round(r["document_probability"], 3),
                        "ai_associated_share": round(r["ai_associated_share"], 3),
                        "human_associated_share": round(r["human_associated_share"], 3),
                        "uncertain_share": round(r["uncertain_share"], 3),
                        "confidence": round(r["confidence"], 3), "category": r["hybrid"]["label"],
                        "evidence": r["evidence_text"]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
