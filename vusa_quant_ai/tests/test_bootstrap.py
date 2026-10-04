"""Installer/launcher logic (bootstrap.py) - Windows behaviour simulated on any OS."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
bootstrap = importlib.import_module("bootstrap")

# Real-world numbers: longest PySide6 file path inside site-packages is 151 characters.
USER_PROJECT = ("C:\\Users\\Χριστοδουλος Γιαγκου\\Desktop\\New folder\\"
                "how-find-your-repository-df3874a628a105694ba995fb6350e17aeefcbfd3\\vusa_quant_ai")


@pytest.fixture
def windows(monkeypatch):
    import shutil
    import tempfile

    short = Path(tempfile.mkdtemp(prefix="b", dir="/tmp" if Path("/tmp").exists() else None))
    monkeypatch.setattr(bootstrap, "IS_WINDOWS", True)
    monkeypatch.setattr(bootstrap, "_windows_long_paths_enabled", lambda: False)
    monkeypatch.setenv("SystemDrive", str(short / "sysdrive"))
    yield short
    shutil.rmtree(short, ignore_errors=True)


def test_user_folder_would_exceed_max_path():
    prefix = USER_PROJECT + "\\.venv\\Lib\\site-packages\\"
    assert len(prefix) + 151 > 259  # why the venv must not live in the project folder


def test_programdata_is_preferred_and_ascii(windows, monkeypatch):
    pd = windows / "ProgramData"
    monkeypatch.setenv("ProgramData", str(pd))
    monkeypatch.setenv("USERPROFILE", str(windows / "Χριστοδουλος"))
    v = bootstrap.choose_venv()
    assert v == pd / "VUSA-Quant" / "venv"
    assert str(v).isascii()


def test_non_ascii_location_is_skipped(windows, monkeypatch):
    monkeypatch.setenv("ProgramData", str(windows / "Δεδομένα"))
    monkeypatch.setenv("USERPROFILE", str(windows / "user"))
    cands = bootstrap.candidate_venvs()
    assert not bootstrap._usable_location(cands[0])  # Greek -> rejected
    assert str(bootstrap.choose_venv()).isascii()


def test_too_long_location_is_skipped(windows, monkeypatch):
    long_base = windows / ("x" * 120)
    assert not bootstrap._usable_location(long_base / "VUSA-Quant" / "venv")


def test_long_paths_enabled_allows_long_location(windows, monkeypatch):
    monkeypatch.setattr(bootstrap, "_windows_long_paths_enabled", lambda: True)
    assert bootstrap._usable_location(windows / ("x" * 120) / "VUSA-Quant" / "venv")


def test_existing_venv_requires_marker(tmp_path, monkeypatch):
    venv = tmp_path / ".venv"
    monkeypatch.setattr(bootstrap, "candidate_venvs", lambda: [venv])
    py = bootstrap.venv_python(venv)
    py.parent.mkdir(parents=True)
    py.write_text("")
    assert bootstrap.existing_venv() is None  # unfinished install -> not used
    (venv / bootstrap.MARKER).write_text("ok")
    assert bootstrap.existing_venv() == venv


def test_broken_venv_is_not_healthy(tmp_path):
    venv = tmp_path / "broken"
    py = bootstrap.venv_python(venv)
    py.parent.mkdir(parents=True)
    py.write_text("not a python")  # e.g. a venv whose base Python was uninstalled
    assert not bootstrap.venv_healthy(venv)


def test_32bit_and_unsupported_python_rejected(monkeypatch):
    monkeypatch.setattr(bootstrap.struct, "calcsize", lambda fmt: 4)
    with pytest.raises(SystemExit, match="32-bit"):
        bootstrap.check_base_python()


def test_torch_failure_does_not_crash_imports(monkeypatch):
    """A half-installed torch (import raises) must leave the app importable with DL marked unavailable."""
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise OSError("[WinError 126] The specified module could not be found")
        return real(name, *a, **k)

    for m in [m for m in sys.modules if m.startswith("vusa_quant_ai.models.deep_learning")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setattr(builtins, "__import__", fake)
    mod = importlib.import_module("vusa_quant_ai.models.deep_learning.models")
    assert mod.TORCH is False and "could not be loaded" in mod._REASON
    assert mod.LSTMForecaster.info.available is False
