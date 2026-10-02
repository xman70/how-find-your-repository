"""CPU / GPU selection. The default "lite" profile never needs a GPU."""
from __future__ import annotations


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch missing or broken: CPU only
        return False


def resolve_device(request: str = "auto") -> tuple[str, str | None]:
    """Return (device, note). ``note`` explains any fallback."""
    request = (request or "auto").lower()
    if request == "cpu":
        return "cpu", None
    has_gpu = cuda_available()
    if request == "gpu":
        if has_gpu:
            return "cuda", None
        return "cpu", "GPU requested but no CUDA device is available; using the CPU."
    return ("cuda" if has_gpu else "cpu"), None
