"""Lightweight JSON experiment tracker (one file per run in experiments/runs/)."""
from __future__ import annotations

import json
import platform
import time
import uuid
from pathlib import Path

from .config import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, float) and o != o:
        return None
    return o


def log_run(kind: str, *, dataset_version: str, feature_version: str, model_version: str, hyperparameters: dict,
            seed: int, validation: dict | None = None, test: dict | None = None, notes: str = "",
            extra: dict | None = None) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    rec = {
        "run_id": run_id, "kind": kind, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": dataset_version, "feature_version": feature_version, "model_version": model_version,
        "hyperparameters": hyperparameters, "seed": seed, "validation": validation or {}, "test": test or {},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "notes": notes,
    }
    if extra:
        rec.update(extra)
    path = RUNS_DIR / f"{run_id}-{kind}.json"
    path.write_text(json.dumps(_jsonable(rec), indent=2), encoding="utf-8")
    return path


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS_DIR.glob("*.json"))]
