"""Five-way human/AI hybrid classification.

0 Human            estimated AI-associated share below ``human_max_share``
1 Human-assisted   human-led text with some AI-associated passages
2 AI-assisted      AI-led text with human contribution
3 AI-generated     estimated AI-associated share above ``ai_assisted_max_share``
4 Unknown          confidence too low or evidence insufficient

The share thresholds are operational *definitions* of the categories (in
config.yaml); the share itself is estimated from calibrated sentence
probabilities and mapped through an isotonic share calibrator fitted on
held-out spliced documents with known AI share.
"""
from __future__ import annotations

from .data import HYBRID_NAMES


def classify(share: float, confidence: float, evidence: str, cfg: dict) -> dict:
    h = cfg.get("hybrid", {})
    if evidence == "insufficient" or confidence < h.get("unknown_below_confidence", 0.35):
        code = 4
        why = ("The text is too short for a reliable estimate." if evidence == "insufficient"
               else "Confidence in the estimate is too low to assign a category.")
    elif share < h.get("human_max_share", 0.15):
        code, why = 0, "Very little of the text shows AI-associated patterns."
    elif share < h.get("human_assisted_max_share", 0.5):
        code, why = 1, "Mostly human-associated patterns with some AI-associated passages."
    elif share < h.get("ai_assisted_max_share", 0.85):
        code, why = 2, "AI-associated patterns dominate, with some human-associated passages."
    else:
        code, why = 3, "Almost all of the text shows AI-associated patterns."
    return {"code": code, "label": HYBRID_NAMES[code], "rationale": why, "estimated_ai_share": float(share)}
