"""Runtime utilities: logging, reproducibility (seeds + environment capture) and
hardware detection (CPU / CUDA) with graceful CPU fallback."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from .. import __version__

TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "xgboost", "lightgbm",
    "catboost", "torch", "hmmlearn", "shap", "optuna", "plotly", "PySide6", "yfinance",
    "feedparser", "requests",
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat(timespec="seconds")


def get_logger(name: str = "vusa") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger("vusa").handlers:
        root = logging.getLogger("vusa")
        root.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(h)
    return logger


def add_file_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("vusa")
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return
    fh = logging.FileHandler(log_dir / "vusa_quant.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # torch optional
        pass


def package_versions() -> dict[str, str]:
    out = {}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


def environment_fingerprint() -> dict:
    return {
        "app_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "captured_at": utcnow_iso(),
    }


def detect_hardware() -> dict:
    info = {"cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count() or 1,
            "ram_gb": None, "gpu": None, "device": "cpu"}
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
        else:  # Windows
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMSTAT()
            m.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            info["ram_gb"] = round(m.ullTotalPhys / 1e9, 1)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["device"] = "cuda"
    except Exception:
        info["torch"] = "not installed - deep learning models will be skipped (reported, not faked)"
    return info


def stable_hash(obj) -> str:
    """Deterministic content hash used for dataset / feature versioning."""
    if hasattr(obj, "to_csv"):
        payload = obj.to_csv().encode()
    else:
        payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
