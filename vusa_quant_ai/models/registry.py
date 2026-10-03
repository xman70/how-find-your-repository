"""Model registry and versioning (champion / challenger, retirement).

Every fitted model gets an immutable ID such as ``XGB_H20_2026_10_03_v17`` together with
training period, feature version, dataset version, hyper-parameters and validation
results. Old versions are never overwritten - retired models keep their history for
benchmarking.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from ..core.runtime import utcnow_iso
from ..database.db import Database
from .deep_learning.models import DL_MODELS
from .statistical.models import STAT_MODELS
from .base import NaiveDrift
from .tree.models import ML_MODELS

ALL_MODELS = {"naive_drift": NaiveDrift, **ML_MODELS, **STAT_MODELS, **DL_MODELS}
SHORT = {"naive_drift": "NAIVE", "elastic_net": "ENET", "random_forest": "RF", "extra_trees": "ET", "hist_gb": "HGB",
         "xgboost": "XGB", "lightgbm": "LGBM", "catboost": "CAT", "arima": "ARIMA", "sarima": "SARIMA",
         "state_space": "STS", "ets": "ETS", "lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "transformer": "TRF",
         "nbeats": "NBEATS", "ensemble": "ENS"}


def model_catalog() -> list[dict]:
    return [{"name": k, "family": v.info.family, "description": v.info.description, "available": v.info.available,
             "reason": v.info.unavailable_reason} for k, v in ALL_MODELS.items()]


class ModelRegistry:
    def __init__(self, db: Database | None, artifact_dir: Path | None):
        self.db, self.dir = db, artifact_dir
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def next_version(self, name: str, horizon: int) -> int:
        if not self.db:
            return 1
        rows = self.db.query("SELECT COUNT(*) AS n FROM model_versions WHERE model_name=? AND horizon=?", [name, horizon])
        return int(rows[0]["n"]) + 1

    def register(self, name: str, horizon: int, train_start, train_end, feature_version: str, dataset_version: str,
                 hyperparameters: dict, validation: dict, model_obj=None, role: str = "candidate") -> str:
        v = self.next_version(name, horizon)
        mid = f"{SHORT.get(name, name.upper())}_H{horizon}_{str(train_end)[:10].replace('-', '_')}_v{v}"
        path = ""
        if model_obj is not None and self.dir is not None:
            try:
                path = str(self.dir / f"{mid}.joblib")
                joblib.dump(model_obj, path)
            except Exception:  # torch modules etc. may not pickle cleanly; metadata still stored
                path = ""
        if self.db:
            self.db.upsert("models", {"model_name": name, "family": ALL_MODELS[name].info.family if name in ALL_MODELS
                                      else "ensemble", "description": ALL_MODELS[name].info.description if name in ALL_MODELS
                                      else "ensemble", "status": "active", "created_at": utcnow_iso()})
            self.db.upsert("model_versions", {
                "model_id": mid, "model_name": name, "horizon": horizon, "train_start": str(train_start)[:10],
                "train_end": str(train_end)[:10], "feature_version": feature_version, "dataset_version": dataset_version,
                "hyperparameters": json.dumps(hyperparameters, default=str), "validation": json.dumps(validation, default=str),
                "artifact_path": path, "role": role, "status": "active", "created_at": utcnow_iso()})
        return mid

    def set_role(self, model_id: str, role: str, status: str | None = None) -> None:
        if not self.db:
            return
        if status:
            self.db.execute("UPDATE model_versions SET role=?, status=? WHERE model_id=?", [role, status, model_id])
        else:
            self.db.execute("UPDATE model_versions SET role=? WHERE model_id=?", [role, model_id])

    def production(self, horizon: int) -> dict | None:
        if not self.db:
            return None
        rows = self.db.query("SELECT * FROM model_versions WHERE horizon=? AND role='production' ORDER BY created_at DESC",
                             [horizon])
        return rows[0] if rows else None

    def versions(self) -> list[dict]:
        return self.db.query("SELECT * FROM model_versions ORDER BY created_at DESC") if self.db else []
